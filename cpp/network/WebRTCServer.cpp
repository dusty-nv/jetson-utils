/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "WebRTCServer.h"
#include "Networking.h"
#include "Process.h"
#include "Thread.h"

#include "json.hpp"
#include "logging.h"

#include <sstream>


// 'video': { width: { ideal: 1280 }, height: { ideal: 720 } }
// HTML outgoing video viewer page template (for server->client)
//  string params:
//    1. websocket path
//    2. websocket protocol (ws or wss)
//    3. stun server
//    4. body content
const char* html_template = " \n \
<html> \n \
  <head> \n \
    <script type='text/javascript' src='https://webrtc.github.io/adapter/adapter-latest.js'></script> \n \
    <script type='text/javascript'> \n \
      var connections = {}; \n \
	 var reportError; \n \
 \n \
      function getLocalStream() { \n \
        var constraints = {'audio': false, 'video': true}; \n \
        if (navigator.mediaDevices.getUserMedia) { \n \
          return navigator.mediaDevices.getUserMedia(constraints); \n \
        } \n \
      } \n \
 \n \
      function onIncomingSDP(url, sdp) { \n \
        console.log('Incoming SDP: (%%s)' + JSON.stringify(sdp), url); \n \
 \n \
        function onLocalDescription(desc) { \n \
          console.log('Local description (%%s)\\n' + JSON.stringify(desc), url); \n \
          connections[url].webrtcPeer.setLocalDescription(desc).then(function() { \n \
            connections[url].websocket.send(JSON.stringify({ type: 'sdp', 'data': connections[url].webrtcPeer.localDescription })); \n \
          }).catch(reportError); \n \
        } \n \
 \n \
        connections[url].webrtcPeer.setRemoteDescription(sdp).catch(reportError); \n \
 \n \
	   if( connections[url].type == 'inbound' ) { \n \
	     connections[url].webrtcPeer.createAnswer().then(onLocalDescription).catch(reportError); \n \
	   } \n \
	   else if( connections[url].type == 'outbound' ) { \n \
	     getLocalStream().then((stream) => { \n \
            console.log('Adding local stream'); \n \
            connections[url].webrtcPeer.addStream(stream); \n \
		  connections[url].webrtcPeer.createAnswer().then(sdp => { \n \
		    // https://stackoverflow.com/a/57674478 \n \
              var arr = sdp.sdp.split('\\r\\n'); \n \
              arr.forEach((str, i) => {  \n \
                if (/^a=fmtp:\\d*/.test(str)) {  \n \
                  arr[i] = str + ';x-google-max-bitrate=10000;x-google-min-bitrate=0;x-google-start-bitrate=6000';  \n \
                } else if (/^a=mid:(1|video)/.test(str)) {  \n \
                  arr[i] += '\\r\\nb=AS:10000';  \n \
                }  \n \
              }); \n \
		    sdp = new RTCSessionDescription({ \n \
                type: 'answer', \n \
                sdp: arr.join('\\r\\n'), \n \
              }); \n \
		    onLocalDescription(sdp); \n \
            }).catch(reportError); \n \
	     }); \n \
        } \n \
 \n \
      } \n \
 \n \
 \n \
      function onIncomingICE(url, ice) { \n \
        var candidate = new RTCIceCandidate(ice); \n \
        console.log('Incoming ICE (%%s)\\n' + JSON.stringify(ice), url); \n \
        connections[url].webrtcPeer.addIceCandidate(candidate).catch(reportError); \n \
      } \n \
 \n \
 \n \
      function getConnectionStats(url, reportType) { \n \
        if( reportType == undefined ) \n \
          reportType = 'all'; \n \
 \n \
        connections[url].webrtcPeer.getStats(null).then((stats) => { \n \
          let statsOutput = ''; \n \
 \n \
          stats.forEach((report) => { \n \
            if( reportType == 'inbound-rtp' && report.type === 'inbound-rtp' && report.kind === 'video') { \n \
		    statsOutput += `# inbound-rtp\n`; \n \
 \n \
              if( connections[url].bytesReceived != undefined ) \n \
                statsOutput += `bitrate:          ${((report.bytesReceived - connections[url].bytesReceived) / 125000).toFixed(3)} mbps\n`; \n \
\n \
              connections[url].bytesReceived = report.bytesReceived; \n \
\n \
		    statsOutput += `bytesReceived:    ${report.bytesReceived}\n`; \n \
              statsOutput += `packetsReceived:  ${report.packetsReceived}\n`; \n \
		    statsOutput += `packetsLost:      ${report.packetsLost}\n`; \n \
              statsOutput += `framesReceived:   ${report.framesReceived}\n`; \n \
              statsOutput += `framesDropped:    ${report.framesDropped}\n`; \n \
              statsOutput += `frameWidth:       ${report.frameWidth}\n`; \n \
              statsOutput += `frameHeight:      ${report.frameHeight}\n`; \n \
              statsOutput += `framesPerSecond:  ${report.framesPerSecond}\n`; \n \
              statsOutput += `keyFramesDecoded: ${report.keyFramesDecoded}\n`; \n \
		    statsOutput += `jitter:           ${report.jitter}\n`; \n \
            } \n \
		  else if( reportType =='outbound-rtp' && report.type === 'outbound-rtp' && report.kind === 'video') { \n \
              statsOutput += `# outbound-rtp\n`; \n \
\n \
              if( connections[url].bytesSent != undefined ) \n \
                statsOutput += `bitrate:          ${((report.bytesSent - connections[url].bytesSent) / 125000).toFixed(3)} mbps\n`; \n \
\n \
              connections[url].bytesSent = report.bytesSent; \n \
\n \
		    statsOutput += `bytesSent:        ${report.bytesSent}\n`; \n \
              statsOutput += `packetsSent:      ${report.packetsSent}\n`; \n \
		    statsOutput += `packetsResent:    ${report.retransmittedPacketsSent}\n`; \n \
              statsOutput += `framesSent:       ${report.framesSent}\n`; \n \
              statsOutput += `frameWidth:       ${report.frameWidth}\n`; \n \
              statsOutput += `frameHeight:      ${report.frameHeight}\n`; \n \
              statsOutput += `framesPerSecond:  ${report.framesPerSecond}\n`; \n \
              statsOutput += `keyFramesSent:    ${report.keyFramesEncoded}\n`; \n \
		  } \n \
            else if( reportType == 'all' || reportType == report.type ) { \n \
              statsOutput += `<h2>Report: ${report.type}</h2>\n<strong>ID:</strong> ${report.id}<br>\n` + \n \
              `<strong>Timestamp:</strong> ${report.timestamp}\n`; \n \
 \n \
              Object.keys(report).forEach((statName) => { \n \
                if (statName !== 'id' && statName !== 'timestamp' && statName !== 'type') \n \
                  statsOutput += `<strong>${statName}:</strong> ${report[statName]}\n`; \n \
              }); \n \
            } \n \
          }); \n \
 \n \
          var statsElement = (connections[url].type == 'inbound') ? 'stats-player' : 'stats-sender'; \n \
          document.getElementById(statsElement).innerHTML = statsOutput; \n \
        }); \n \
      } \n \
 \n \
 \n \
      function onAddRemoteStream(event) { \n \
	   var url = event.srcElement.url; \n \
	   console.log('Adding remote stream to HTML video player (%%s)', url); \n \
        connections[url].videoElement.srcObject = event.streams[0]; \n \
	   connections[url].videoElement.play(); \n \
      } \n \
 \n \
 \n \
      function onIceCandidate(event) { \n \
	   var url = event.srcElement.url; \n \
 \n \
        if (event.candidate == null) \n \
          return; \n \
 \n \
        console.log('Sending ICE candidate out (%%s)\\n' + JSON.stringify(event.candidate), url); \n \
        connections[url].websocket.send(JSON.stringify({'type': 'ice', 'data': event.candidate })); \n \
      } \n \
 \n \
 \n \
      function onServerMessage(event) { \n \
        var msg; \n \
	   var url = event.srcElement.url; \n \
 \n \
        try { \n \
          msg = JSON.parse(event.data); \n \
        } catch (e) { \n \
          return; \n \
        } \n \
 \n \
        if( !connections[url].webrtcPeer ) { \n \
          connections[url].webrtcPeer = new RTCPeerConnection(connections[url].webrtcConfig); \n \
		connections[url].webrtcPeer.url = url; \n \
 \n \
		connections[url].webrtcPeer.onconnectionstatechange = (ev) => { \n \
	       console.log('WebRTC connection state (%%s) ' + connections[url].webrtcPeer.connectionState, url); \n \
            if( connections[url].webrtcPeer.connectionState == 'connected' ) \n \
              setInterval(getConnectionStats, 1000, url, connections[url].type == 'inbound' ? 'inbound-rtp' : 'outbound-rtp'); \n \
	     } \n \
 \n \
          if( connections[url].type == 'inbound' ) \n \
            connections[url].webrtcPeer.ontrack = onAddRemoteStream; \n \
          connections[url].webrtcPeer.onicecandidate = onIceCandidate; \n \
        } \n \
 \n \
        switch (msg.type) { \n \
          case 'sdp': onIncomingSDP(url, msg.data); break; \n \
          case 'ice': onIncomingICE(url, msg.data); break; \n \
          default: break; \n \
        } \n \
      } \n \
 \n \
 \n \
      function playStream(videoPlayer, hostname, port, path, configuration, reportErrorCB) { \n \
        var l = window.location;\n \
        if( path == 'null' ) \n \
	      return; \n \
        var wsProt = (l.protocol == 'https:') ? 'wss://' : 'ws://'; \n \
        var wsHost = (hostname != undefined) ? hostname : l.hostname; \n \
        var wsPort = (port != undefined) ? port : l.port; \n \
        var wsPath = (path != undefined) ? path : '/ws'; \n \
        if (wsPort) \n\
          wsPort = ':' + wsPort; \n\
        var wsUrl = wsProt + wsHost + wsPort + wsPath; \n \
	   console.log('Video server URL: ' + wsUrl); \n \
	   var url = wsUrl; \n \
 \n \
        connections[url] = {}; \n \
 \n \
        connections[url].type = 'inbound'; \n \
        connections[url].videoElement = document.getElementById(videoPlayer); \n \
        connections[url].webrtcConfig = configuration; \n \
        reportError = (reportErrorCB != undefined) ? reportErrorCB : function(text) {}; \n \
 \n \
        connections[url].websocket = new WebSocket(wsUrl); \n \
        connections[url].websocket.addEventListener('message', onServerMessage); \n \
      } \n \
 \n \
      function sendStream(hostname, port, path, configuration, reportErrorCB) { \n \
        var l = window.location; \n \
        if( path == 'null' ) \n \
	      return; \n \
        if( l.protocol != 'https:' ) { \n \
	      alert('Please use HTTPS to enable the use of your browser webcam'); \n \
		 return; \n \
	   } \n \
        if( !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia ) { \n \
	      alert('getUserMedia() not available (confirm HTTPS is being used)'); \n \
		 return; \n \
        } \n \
	   var wsProt = (l.protocol == 'https:') ? 'wss://' : 'ws://'; \n \
        var wsHost = (hostname != undefined) ? hostname : l.hostname; \n \
        var wsPort = (port != undefined) ? port : l.port; \n \
        var wsPath = (path != undefined) ? path : '/ws'; \n \
        if (wsPort) \n\
          wsPort = ':' + wsPort; \n\
        var wsUrl = wsProt + wsHost + wsPort + wsPath; \n \
        console.log('Video server URL: ' + wsUrl); \n \
	   var url = wsUrl; \n \
 \n \
        connections[url] = {}; \n \
 \n \
        connections[url].type = 'outbound'; \n \
        connections[url].webrtcConfig = configuration; \n \
        reportError = (reportErrorCB != undefined) ? reportErrorCB : function(text) {}; \n \
 \n \
        connections[url].websocket = new WebSocket(wsUrl); \n \
        connections[url].websocket.addEventListener('message', onServerMessage); \n \
      } \n \
 \n \
      window.onload = function() { \n \
        var config = { 'iceServers': [%s] }; \n\
        playStream('video-player', null, null, '%s', config, function (errmsg) { console.error(errmsg); }); \n \
        sendStream(null, null, '%s', config, function (errmsg) { console.error(errmsg); }); \n \
	 }; \n \
 \n \
    </script> \n \
  </head> \n \
 \n \
  <body style='background-color:#333333; color:#FFFFFF;'> \n \
    <div> \n \
      <video id='video-player' autoplay controls playsinline muted>Your browser does not support video</video> \n \
    </div> \n \
    <pre>%s</pre> \n \
    <pre id='stats-player'></pre> \n \
    <pre id='stats-sender'></pre> \n \
  </body> \n \
</html> \n \
";

// when the requested stream couldn't be found
const char* html_not_found = " \n \
<html> \n \
<body> \n \
    <p>Couldn't find stream %s</p>\n \
  </body> \n \
</html> \n \
";

// when the server doesn't have a page for this type of stream
const char* html_unable = " \n \
<html> \n \
<body> \n \
    <p>Couldn't handle this type of stream: %s</p>\n \
  </body> \n \
</html> \n \
";

// when there are no streams on the server
const char* html_inactive = " \n \
<html> \n \
<body> \n \
    <p>No streams</p>\n \
  </body> \n \
</html> \n \
";



// remove an element from a vector (only removes the first instance)
template<typename T> static void vector_remove_element( std::vector<T>& vector, const T& element )
{
	const size_t numElements = vector.size();
	
	for( size_t n=0; n < numElements; n++ )
	{
		if( vector[n] == element )
		{
			vector.erase(vector.begin() + n);
			return;
		}
	}
}


// list of existing server instances
std::vector<WebRTCServer*> gWebRTCServers;


// constructor
WebRTCServer::WebRTCServer( uint16_t port, const char* stun_server, const char* ssl_cert_file, const char* ssl_key_file, bool threaded )
{	
	mPort = port;
	mRefCount = 1;
	mPeerCount = 0;
	mHasHTTPS = false;
	mSoupServer = NULL;
	
	if( stun_server != NULL )
		mStunServer = stun_server;
	
	if( ssl_cert_file != NULL )
		mSSLCertFile = ssl_cert_file;
	
	if( ssl_key_file != NULL )
		mSSLKeyFile = ssl_key_file;
	
	if( threaded )
		mThread = new Thread();
}


// destructor
WebRTCServer::~WebRTCServer()
{
	// TODO free routes and peers
	/*if( mWebsocketConnection != NULL )
	{
		g_object_unref(mWebsocketConnection);
		mWebsocketConnection = NULL;
	}*/
	
	if( mThread != NULL )
	{
		//mThread->Stop(true);
		delete mThread;
		mThread = NULL;
	}

	if( mSoupServer != NULL )
	{
		g_object_unref(mSoupServer);
		mSoupServer = NULL;
	}
}


// Release
void WebRTCServer::Release()
{
	mRefCount--;
	
	if( mRefCount == 0 )
	{
		LogInfo(LOG_WEBRTC "WebRTC server on port %hu is shutting down\n", mPort);
		vector_remove_element(gWebRTCServers, this);
		delete this;
	}
}
		

// Create
WebRTCServer* WebRTCServer::Create( uint16_t port, const char* stun_server, const char* ssl_cert_file, const char* ssl_key_file, bool threaded )
{
	// see if a server on this port already exists
	const uint32_t numServers = gWebRTCServers.size();
	
	for( uint32_t n=0; n < numServers; n++ )
	{
		if( gWebRTCServers[n]->mPort == port )
		{
			gWebRTCServers[n]->mRefCount++;
			return gWebRTCServers[n];
		}
	}
	
	// assign a default STUN server if needed
	if( !stun_server || strlen(stun_server) == 0 ) 
	{
		stun_server = WEBRTC_DEFAULT_STUN_SERVER;
    }
	else if( strcasecmp(stun_server, "disable") == 0 || strcasecmp(stun_server, "disabled") == 0 || strcasecmp(stun_server, "off") == 0 ) 
	{
	    stun_server = NULL;
	}
	    
	// create a new server
	WebRTCServer* server = new WebRTCServer(port, stun_server, ssl_cert_file, ssl_key_file, threaded);

	if( !server || !server->init() )
	{
		LogError(LOG_WEBRTC "failed to create WebRTC server on port %hu\n", port);
		return NULL;
	}
	
	// start the thread if needed
	if( server->mThread != NULL )
	{
		if( !server->mThread->Start(runThread, server) )
		{
			LogError(LOG_WEBRTC "failed to start thread for running WebRTC server\n");
			return NULL;
		}
	}
	
	gWebRTCServers.push_back(server);
	return server;
}


// init
bool WebRTCServer::init()
{
	// create the soup server
	mSoupServer = soup_server_new(SOUP_SERVER_SERVER_HEADER, "webrtc-server", NULL);
	
	if( !mSoupServer )
	{
		LogError(LOG_WEBRTC "failed to create SOUP server\n");
		return false;
	}
								
	// load SSL/HTTPS certificate
	if( mSSLCertFile.length() > 0 && mSSLKeyFile.length() > 0 )
	{
		GError* error = NULL;
		
		if( !soup_server_set_ssl_cert_file(mSoupServer, mSSLCertFile.c_str(), mSSLKeyFile.c_str(), &error) )
		{
			LogError(LOG_WEBRTC "failed to load SSL certificate, unable to use HTTPS\n");
			LogError(LOG_WEBRTC "(%s)\n", error->message);
			g_error_free(error);	
			return false;
		}

		mHasHTTPS = true;
	}
	else if( mSSLCertFile.length() > 0 || mSSLKeyFile.length() > 0 )
	{
		LogError(LOG_WEBRTC "must provide valid SSL certificate AND key files to enable HTTPS\n");
		LogError(LOG_WEBRTC "(see the --ssl-cert and --ssl-key command line options)\n");
		return false;
	}
	
	// add default handlers
	soup_server_add_handler(mSoupServer, "/", onHttpRequest, this, NULL);
	//soup_server_add_websocket_handler(mSoupServer, "/", NULL, NULL, onWebsocketOpened, this, NULL);
	
	AddRoute("/", onHttpDefault, this);  // serve the server-default HTML pages
	
	// start the server listening
	GError* err = NULL;

	if( !soup_server_listen_all(mSoupServer, mPort, mHasHTTPS ? SOUP_SERVER_LISTEN_HTTPS : (SoupServerListenOptions)0, &err) )
	{
		LogError(LOG_WEBRTC "SOUP server failed to listen on port %hu\n", mPort);
		LogError(LOG_WEBRTC "   (%s)\n", err->message);
		g_error_free(err);
	}
	
	LogSuccess(LOG_WEBRTC "WebRTC server started @ %s://%s:%hu\n", mHasHTTPS ? "https" : "http", getHostname().c_str(), mPort);
	return true;
}


// Add http route
void WebRTCServer::AddRoute( const char* path, WebRTCServer::HttpListener callback, void* user_data, uint32_t flags )
{
	// root clears all existing routes
	if( strcmp(path, "/") == 0 )
	{
		const uint32_t numRoutes = mHttpRoutes.size();
	
		for( uint32_t n=0; n < numRoutes; n++ )
			freeRoute(mHttpRoutes[n]);
		
		mHttpRoutes.clear();
	}
	
	if( !callback )
		return;
	
	// if there was an existing root route, remove it
	if( mHttpRoutes.size() == 1 && mHttpRoutes[0]->path == "/" )
	{
		freeRoute(mHttpRoutes[0]);
		mHttpRoutes.clear();
	}
	
	// create a new route
	HttpRoute* route = new HttpRoute();
	
	route->path = path;
	route->callback = callback;
	route->user_data = user_data;
	route->flags = flags;
			
	// if this route already exists, replace it
	const uint32_t numRoutes = mHttpRoutes.size();
	
	for( uint32_t n=0; n < numRoutes; n++ )
	{
		if( mHttpRoutes[n]->path == path )
		{
			freeRoute(mHttpRoutes[n]);
			mHttpRoutes[n] = route;
			return;
		}
	}
	
	mHttpRoutes.push_back(route);
	return;
}


// Add websocket route
void WebRTCServer::AddRoute( const char* path, WebRTCServer::WebsocketListener callback, void* user_data, uint32_t flags )
{
	// root clears all existing routes
	if( strcmp(path, "/") == 0 )
	{
		const uint32_t numRoutes = mWebsocketRoutes.size();
	
		for( uint32_t n=0; n < numRoutes; n++ )
			freeRoute(mWebsocketRoutes[n]);
		
		mWebsocketRoutes.clear();
	}
	
	if( !callback )
		return;
	
	// create a new route
	WebsocketRoute* route = new WebsocketRoute();
	
	route->path = path;
	route->callback = callback;
	route->user_data = user_data;
	route->flags = flags;
			
	// if this route already exists, replace it
	const uint32_t numRoutes = mWebsocketRoutes.size();
	
	for( uint32_t n=0; n < numRoutes; n++ )
	{
		if( mWebsocketRoutes[n]->path == path )
		{
			freeRoute(mWebsocketRoutes[n]);
			mWebsocketRoutes[n] = route;
			return;
		}
	}
	
	mWebsocketRoutes.push_back(route);
	LogVerbose(LOG_WEBRTC "websocket route added %s\n", path);
	
	soup_server_add_websocket_handler(mSoupServer, path, NULL, NULL, onWebsocketOpened, this, NULL);
	return;
}


// findHttpRoute
WebRTCServer::HttpRoute* WebRTCServer::findHttpRoute( const char* path ) const
{
	if( !path )
		return NULL;
	
	const uint32_t numRoutes = mHttpRoutes.size();
	
	for( uint32_t n=0; n < numRoutes; n++ )
	{
		if( mHttpRoutes[n]->path == path )
			return mHttpRoutes[n];
	}
	
	return NULL;
}


// findWebsocketRoute
WebRTCServer::WebsocketRoute* WebRTCServer::findWebsocketRoute( const char* path ) const
{
	if( !path )
		return NULL;
	
	const uint32_t numRoutes = mWebsocketRoutes.size();
	
	for( uint32_t n=0; n < numRoutes; n++ )
	{
		if( mWebsocketRoutes[n]->path == path )
			return mWebsocketRoutes[n];
	}
	
	return NULL;
}


// freeRoute
void WebRTCServer::freeRoute( WebRTCServer::HttpRoute* route )
{
	delete route;
}


// freeRoute
void WebRTCServer::freeRoute( WebRTCServer::WebsocketRoute* route )
{
	const size_t numPeers = route->peers.size();
	
	for( size_t n=0; n < numPeers; n++ )
		delete route->peers[n];
	
	delete route;
}


// onHttpRequest
void WebRTCServer::onHttpRequest( SoupServer* soup_server, SoupMessage* message, const char* path, GHashTable* query, SoupClientContext* client_context, void* user_data )
{
	WebRTCServer* server = (WebRTCServer*)user_data;
	
	if( !server )
	{
		soup_message_set_status(message, SOUP_STATUS_INTERNAL_SERVER_ERROR);
		return;
	}
	
	// find if path is found
	HttpRoute* route = server->findHttpRoute(path);
	
	if( !route )
	{
		if( server->mHttpRoutes.size() == 1 && server->mHttpRoutes[0]->path == "/" )
			route = server->mHttpRoutes[0];
	}

	if( !route )
	{
		LogVerbose(LOG_WEBRTC "%s %s %s '%s' -- not found 404\n", server->HasHTTPS() ? "HTTPS" : "HTTP", soup_client_context_get_host(client_context), message->method, path);
		soup_message_set_status(message, SOUP_STATUS_NOT_FOUND);
		return;
	}
	
	LogVerbose(LOG_WEBRTC "%s %s %s '%s'\n", server->HasHTTPS() ? "HTTPS" : "HTTP", soup_client_context_get_host(client_context), message->method, path);
	
	// dispatch callback
	route->callback(soup_server, message, path, query, client_context, route->user_data);
}

	
// onHttpDefault (this serves the default site)
void WebRTCServer::onHttpDefault( SoupServer* soup_server, SoupMessage* message, const char* path, GHashTable* query, SoupClientContext* client_context, void* user_data )
{
	WebRTCServer* server = (WebRTCServer*)user_data;
	
	if( !server )
	{
		soup_message_set_status(message, SOUP_STATUS_INTERNAL_SERVER_ERROR);
		return;
	}
	
	LogVerbose(LOG_WEBRTC "%s\n", path);
	
	// JSON REST API
	if( strcmp(path, "/api/streams") == 0 )
	{
		nlohmann::json json;
		
		const uint32_t numWebsocketRoutes = server->mWebsocketRoutes.size();
		
		for( uint32_t n=0; n < numWebsocketRoutes; n++ )
		{
			WebsocketRoute* route = server->mWebsocketRoutes[n];
			
			std::vector<std::string> flags;
			
			if( route->flags & WEBRTC_AUDIO )  
				flags.push_back("audio");
			
			if( route->flags & WEBRTC_VIDEO )
				flags.push_back("video");
			
			if( route->flags & WEBRTC_SEND )
				flags.push_back("send");
			
			if( route->flags & WEBRTC_RECEIVE )
				flags.push_back("receive");
			
			if( route->flags & WEBRTC_MULTI_CLIENT )
				flags.push_back("multi_client");
			
			json[route->path]["flags"] = flags;
			json[route->path]["peer_count"] = route->peers.size();
		}
		
		const std::string json_str = json.dump(2);
		
		// reply with the JSON
		SoupBuffer* soup_buffer = soup_buffer_new(SOUP_MEMORY_COPY, json_str.c_str(), json_str.length()); // SOUP_MEMORY_STATIC

		soup_message_headers_set_content_type(message->response_headers, "application/json", NULL);
		soup_message_body_append_buffer(message->response_body, soup_buffer);
		soup_buffer_free(soup_buffer);

		soup_message_set_status(message, SOUP_STATUS_OK);
		return;
	}

	// the HTML to serve will be rendered to this buffer
	char html[16384];

	#define CHECK_SNPRINTF(x) \
		const int chars_needed = x; \
		if( chars_needed < 0 || chars_needed >= sizeof(html) ) { \
			LogError(LOG_WEBRTC "buffer length exceeded rendering html template (%i vs %zu bytes)\n", chars_needed, sizeof(html)); \
			soup_message_set_status(message, SOUP_STATUS_INTERNAL_SERVER_ERROR); \
			return; \
		}
				
	// get the stream name the user wishes to view from the 'stream' query param
	// this takes the form:  http://0.0.0.0:8080/?stream=name  (or any page, like index.html?stream=name)
	// it's done with query params to avoid collisions with the websockets running on this port
	// if 'stream' isn't specified in the URL, then it will default to the first available stream
	const char* stream = NULL;
	
	if( query != NULL )
		stream = (const char*)g_hash_table_lookup(query, "stream");

	if( !stream )
	{
		for( uint32_t n=0; n < server->mWebsocketRoutes.size(); n++ )
		{
			stream = server->mWebsocketRoutes[n]->path.c_str();
			
			if( (server->mWebsocketRoutes[n]->flags & WEBRTC_VIDEO) && (server->mWebsocketRoutes[n]->flags & WEBRTC_SEND) )
				break; // if there are any playback streams, default to the first one
		}
	}
	
	if( stream != NULL )
	{
		WebsocketRoute* route = server->findWebsocketRoute(stream);
		
		if( !route && stream[0] != '/' )  // append '/' to stream name if needed
		{
			const std::string stream_leading_slash = std::string("/") + std::string(stream);
			route = server->findWebsocketRoute(stream_leading_slash.c_str());
		}
		
		if( route != NULL )
		{
			const std::string stream_info = server->printRouteInfo(route);
			
			const char* send_path = "null";
			const char* receive_path = "null";
			
			if( (route->flags & WEBRTC_VIDEO) && (route->flags & WEBRTC_SEND) )
			{
				send_path = route->path.c_str();
				
				for( uint32_t n=0; n < server->mWebsocketRoutes.size(); n++ )
				{
					if( (server->mWebsocketRoutes[n]->flags & WEBRTC_VIDEO) && (server->mWebsocketRoutes[n]->flags & WEBRTC_RECEIVE) )
					{
						receive_path = server->mWebsocketRoutes[n]->path.c_str();
						break;
					}
				}
			}
			else if( (route->flags & WEBRTC_VIDEO) && (route->flags & WEBRTC_RECEIVE) )
			{
				receive_path = route->path.c_str();
			}
			
			char stun_str[1024];
			stun_str[0] = '\0';
			
			const char* stun_server = server->GetSTUNServer();
			
			if( stun_server != NULL && strlen(stun_server) > 0 )
			    snprintf(stun_str, sizeof(stun_str), "{ 'urls': 'stun:%s' }", stun_server);
               
			CHECK_SNPRINTF(snprintf(html, sizeof(html), html_template, 
								    stun_str, send_path, receive_path,
								    stream_info.c_str()));
		}
		else
		{
			CHECK_SNPRINTF(snprintf(html, sizeof(html), html_not_found, stream));
		}
	}
	else
	{
		strncpy(html, html_inactive, sizeof(html));	// no streams available
	}

	// reply with the HTML content
	SoupBuffer* soup_buffer = soup_buffer_new(SOUP_MEMORY_COPY, html, strlen(html)); // SOUP_MEMORY_STATIC

	soup_message_headers_set_content_type(message->response_headers, "text/html", NULL);
	soup_message_body_append_buffer(message->response_body, soup_buffer);
	soup_buffer_free(soup_buffer);

	soup_message_set_status(message, SOUP_STATUS_OK);
}


// print stream info for use in HTML
std::string WebRTCServer::printRouteInfo( WebsocketRoute* route ) const
{
	std::ostringstream ss;
	
	ss << "<p>Stream " << route->path << "&nbsp;&nbsp;&nbsp;(flags:";

	if( route->flags & WEBRTC_AUDIO )
		ss << " audio";
	
	if( route->flags & WEBRTC_VIDEO )
		ss << " video";
	
	if( route->flags & WEBRTC_SEND )
		ss << " send";
	
	if( route->flags & WEBRTC_RECEIVE )
		ss << " receive";
	
	if( route->flags & WEBRTC_MULTI_CLIENT )
		ss << " multi-client";
	
	ss << ")&nbsp;&nbsp;(peers: " << route->peers.size() << ")</p>";
	ss << "<p>" << Process::GetCommandLine() << "</p>";
	
	return ss.str();
}


// onWebsocketOpened
void WebRTCServer::onWebsocketOpened( SoupServer* soup_server, SoupWebsocketConnection* connection, const char *path, SoupClientContext* client_context, void* user_data )
{	
	WebRTCServer* server = (WebRTCServer*)user_data;
	
	if( !server )
		return;
	
	const char* ip_address = soup_client_context_get_host(client_context);
	LogInfo(LOG_WEBRTC "websocket %s -- new connection opened by %s (peer_id=%u)\n", path, ip_address, server->mPeerCount);
	
	// lookup the route using the path the websocket connected on
	WebsocketRoute* route = server->findWebsocketRoute(path);
	
	if( !route )
	{
		LogVerbose(LOG_WEBRTC "websocket %s %s not found\n", ip_address, path);
		return;
	}
	
	// create new peer object
	WebRTCPeer* peer = new WebRTCPeer();
		
	peer->connection = connection;
	peer->client_context = client_context;
	peer->server = server;
	
	peer->ID = server->mPeerCount;
	peer->path = path;
	peer->flags = route->flags | WEBRTC_PEER_CONNECTING;
	peer->user_data = NULL;
	peer->ip_address = ip_address;
	
	route->peers.push_back(peer);
	server->mPeerCount++;
		
	g_object_ref(G_OBJECT(connection));
	
	// subscribe to messages
	g_signal_connect(G_OBJECT(connection), "message", G_CALLBACK(onWebsocketMessage), peer);
	g_signal_connect(G_OBJECT(connection), "closed", G_CALLBACK(onWebsocketClosed), peer);

	// call the route
	route->callback(peer, NULL, 0, route->user_data);
	
	// update flags
	peer->flags &= ~WEBRTC_PEER_CONNECTING;
	peer->flags |= WEBRTC_PEER_CONNECTED;
}
	
	
// onWebsocketClosed	
void WebRTCServer::onWebsocketClosed( SoupWebsocketConnection* connection, void* user_data )
{
	if( !user_data )
		return;
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;
	
	LogInfo(LOG_WEBRTC "websocket %s -- connection to %s (peer_id=%u) closed\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID);
	
	peer->flags &= ~(WEBRTC_PEER_CONNECTED|WEBRTC_PEER_STREAMING);
	peer->flags |= WEBRTC_PEER_CLOSED;
	
	WebsocketRoute* route = peer->server->findWebsocketRoute(peer->path.c_str());

	if( route != NULL )
	{
		route->callback(peer, NULL, 0, route->user_data); // closed flag set above
		vector_remove_element(route->peers, peer);
	}
	
	g_object_unref(G_OBJECT(peer->connection));
	delete peer;
}


// onWebsocketMessage
void WebRTCServer::onWebsocketMessage( SoupWebsocketConnection* connection, SoupWebsocketDataType data_type, GBytes* message, void* user_data )
{
	if( !user_data )
		return;
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;

	LogVerbose(LOG_WEBRTC "websocket %s -- recieved message from %s (peer_id=%u) (%zu bytes)\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, g_bytes_get_size(message));
	
	// extract the message to string
	gchar* data = NULL;
	gchar* data_string = NULL;
	gsize  data_size = 0;
	
	switch (data_type) 
	{
		case SOUP_WEBSOCKET_DATA_BINARY:
			LogWarning(LOG_WEBRTC "websocket %s received unknown binary message from %s, ignoring\n", peer->path.c_str(), peer->ip_address.c_str());
			g_bytes_unref(message);
			return;

		case SOUP_WEBSOCKET_DATA_TEXT:
			data = (gchar*)g_bytes_unref_to_data(message, &data_size);
			data_string = g_strndup(data, data_size); // Convert to NULL-terminated string
			g_free(data);
			break;

		default:
			g_assert_not_reached();
	}

	// relay to route handler
	WebsocketRoute* route = peer->server->findWebsocketRoute(peer->path.c_str());

	if( route != NULL )
		route->callback(peer, data_string, data_size, route->user_data);
		
	g_free(data_string);
}



// ProcessRequests
bool WebRTCServer::ProcessRequests( bool blocking )
{
	// https://stackoverflow.com/questions/23737750/glib-usage-without-mainloop
	// https://www.freedesktop.org/software/gstreamer-sdk/data/docs/2012.5/glib/glib-The-Main-Event-Loop.html#g-main-context-iteration
	// https://developer-old.gnome.org/programming-guidelines/stable/main-contexts.html.en
	g_main_context_iteration(NULL, blocking);
	return true;
}


// runThread
void* WebRTCServer::runThread( void* user_data )
{
	WebRTCServer* server = (WebRTCServer*)user_data;
	
	if( !server )
		return 0;
	
	LogVerbose(LOG_WEBRTC "WebRTC server thread running...\n");
	
	while( server->mRefCount > 0 )
	{
		//LogDebug(LOG_WEBRTC "WebRTC processing requests\n");
		server->ProcessRequests(true);
	}
	
	LogVerbose(LOG_WEBRTC "WebRTC server thread stopped\n");
	return 0;
}

	
