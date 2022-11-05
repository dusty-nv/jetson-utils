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

#include "NetworkAdapter.h"
#include "logging.h"


#define LOG_WEBRTC "[webrtc] "


// TODO make this configurable
#define STUN_SERVER "stun.l.google.com:19302"

const char* html_viewer = " \n \
<html> \n \
  <head> \n \
    <script type=\"text/javascript\" src=\"https://webrtc.github.io/adapter/adapter-latest.js\"></script> \n \
    <script type=\"text/javascript\"> \n \
      var html5VideoElement; \n \
      var websocketConnection; \n \
      var webrtcPeerConnection; \n \
      var webrtcConfiguration; \n \
      var reportError; \n \
 \n \
 \n \
      function onLocalDescription(desc) { \n \
        console.log(\"Local description: \" + JSON.stringify(desc)); \n \
        webrtcPeerConnection.setLocalDescription(desc).then(function() { \n \
          websocketConnection.send(JSON.stringify({ type: \"sdp\", \"data\": webrtcPeerConnection.localDescription })); \n \
        }).catch(reportError); \n \
      } \n \
 \n \
 \n \
      function onIncomingSDP(sdp) { \n \
        console.log(\"Incoming SDP: \" + JSON.stringify(sdp)); \n \
        webrtcPeerConnection.setRemoteDescription(sdp).catch(reportError); \n \
        webrtcPeerConnection.createAnswer().then(onLocalDescription).catch(reportError); \n \
      } \n \
 \n \
 \n \
      function onIncomingICE(ice) { \n \
        var candidate = new RTCIceCandidate(ice); \n \
        console.log(\"Incoming ICE: \" + JSON.stringify(ice)); \n \
        webrtcPeerConnection.addIceCandidate(candidate).catch(reportError); \n \
      } \n \
 \n \
 \n \
      function onAddRemoteStream(event) { \n \
        html5VideoElement.srcObject = event.streams[0]; \n \
      } \n \
 \n \
 \n \
      function onIceCandidate(event) { \n \
        if (event.candidate == null) \n \
          return; \n \
 \n \
        console.log(\"Sending ICE candidate out: \" + JSON.stringify(event.candidate)); \n \
        websocketConnection.send(JSON.stringify({ \"type\": \"ice\", \"data\": event.candidate })); \n \
      } \n \
 \n \
 \n \
      function onServerMessage(event) { \n \
        var msg; \n \
 \n \
        try { \n \
          msg = JSON.parse(event.data); \n \
        } catch (e) { \n \
          return; \n \
        } \n \
 \n \
        if (!webrtcPeerConnection) { \n \
          webrtcPeerConnection = new RTCPeerConnection(webrtcConfiguration); \n \
          webrtcPeerConnection.ontrack = onAddRemoteStream; \n \
          webrtcPeerConnection.onicecandidate = onIceCandidate; \n \
        } \n \
 \n \
        switch (msg.type) { \n \
          case \"sdp\": onIncomingSDP(msg.data); break; \n \
          case \"ice\": onIncomingICE(msg.data); break; \n \
          default: break; \n \
        } \n \
      } \n \
 \n \
 \n \
      function playStream(videoElement, hostname, port, path, configuration, reportErrorCB) { \n \
        var l = window.location;\n \
        var wsHost = (hostname != undefined) ? hostname : l.hostname; \n \
        var wsPort = (port != undefined) ? port : l.port; \n \
        var wsPath = (path != undefined) ? path : \"ws\"; \n \
        if (wsPort) \n\
          wsPort = \":\" + wsPort; \n\
        var wsUrl = \"ws://\" + wsHost + wsPort + \"/\" + wsPath; \n \
 \n \
        html5VideoElement = videoElement; \n \
        webrtcConfiguration = configuration; \n \
        reportError = (reportErrorCB != undefined) ? reportErrorCB : function(text) {}; \n \
 \n \
        websocketConnection = new WebSocket(wsUrl); \n \
        websocketConnection.addEventListener(\"message\", onServerMessage); \n \
      } \n \
 \n \
      window.onload = function() { \n \
        var vidstream = document.getElementById(\"stream\"); \n \
        var config = { 'iceServers': [{ 'urls': 'stun:" STUN_SERVER "' }] }; \n\
        playStream(vidstream, null, null, null, config, function (errmsg) { console.error(errmsg); }); \n \
      }; \n \
 \n \
    </script> \n \
  </head> \n \
 \n \
  <body> \n \
    <div> \n \
      <video id=\"stream\" autoplay controls playsinline>Your browser does not support video</video> \n \
    </div> \n \
  </body> \n \
</html> \n \
";

const char* html_inactive = " \n \
<html> \n \
<body> \n \
    <p>No streams</p>\n \
  </body> \n \
</html> \n \
";



// constructor
WebRTCServer::WebRTCServer( uint16_t port )
{	
	mPort = port;
	mRefCount = 1;
	mSoupServer = NULL;
}


// destructor
WebRTCServer::~WebRTCServer()
{
	/*if( mWebsocketConnection != NULL )
	{
		g_object_unref(mWebsocketConnection);
		mWebsocketConnection = NULL;
	}*/

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
		delete this;
}
		

// Create
WebRTCServer* WebRTCServer::Create( uint16_t port )
{
	WebRTCServer* server = new WebRTCServer(port);

	if( !server || !server->init() )
	{
		LogError(LOG_WEBRTC "failed to create WebRTC server on port %hu\n", port);
		return NULL;
	}
	
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
	
	soup_server_add_handler(mSoupServer, "/", onHttpRequest, this, NULL);
	//soup_server_add_websocket_handler(mSoupServer, "/ws", NULL, NULL, onWebsocketConnection, this, NULL);
	AddRoute("/", onHttpDefault, this);
	
	// start the server listening
	GError* err = NULL;

	if( !soup_server_listen_all(mSoupServer, mPort, (SoupServerListenOptions)0, &err) )
	{
		LogError(LOG_WEBRTC "SOUP server failed to listen on port %hu\n", mPort);
		LogError(LOG_WEBRTC "   (%s)\n", err->message);
		g_error_free(err);
	}
	
	LogSuccess(LOG_WEBRTC "WebRTC server started @ http://%s:%hu\n", networkHostname().c_str(), mPort);
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
	if( !user_data )
	{
		soup_message_set_status(message, SOUP_STATUS_INTERNAL_SERVER_ERROR);
		return;
	}
	
	WebRTCServer* server = (WebRTCServer*)user_data;
	
	// find if path is found
	HttpRoute* route = server->findHttpRoute(path);
	
	if( !route )
	{
		LogVerbose(LOG_WEBRTC "HTTP %s %s '%s' -- not found 404\n", soup_client_context_get_host(client_context), message->method, path);
		soup_message_set_status(message, SOUP_STATUS_NOT_FOUND);
		return;
	}
	
	LogVerbose(LOG_WEBRTC "HTTP %s %s '%s'\n", soup_client_context_get_host(client_context), message->method, path);
	
	// dispatch callback
	route->callback(soup_server, message, path, query, client_context, route->user_data);
}


// onHttpDefault (this serves the default site)
void WebRTCServer::onHttpDefault( SoupServer* soup_server, SoupMessage* message, const char* path, GHashTable* query, SoupClientContext* client_context, void* user_data )
{
	if( !user_data )
	{
		soup_message_set_status(message, SOUP_STATUS_INTERNAL_SERVER_ERROR);
		return;
	}
	
	WebRTCServer* server = (WebRTCServer*)user_data;
	const char* html = NULL;
	
	// find if path is found
	WebsocketRoute* websocketRoute = server->findWebsocketRoute(path);
	
	if( !websocketRoute )
	{
		if( strcmp(path, "/") == 0 || strcmp(path, "index.html") == 0 )
			html = html_inactive;
	}
	
	if( !html )
	{
		soup_message_set_status(message, SOUP_STATUS_NOT_FOUND);
		return;
	}
		
	SoupBuffer* soup_buffer = soup_buffer_new(SOUP_MEMORY_STATIC, html, strlen(html)); // TODO watch SOUP_MEMORY_STATIC when we start making other strings

	soup_message_headers_set_content_type(message->response_headers, "text/html", NULL);
	soup_message_body_append_buffer(message->response_body, soup_buffer);
	soup_buffer_free(soup_buffer);

	soup_message_set_status(message, SOUP_STATUS_OK);
}


/*
// onWebsocketConnection
void WebRTCServer::onWebsocketConnection( SoupServer* soup_server, SoupMessage* message, const char* path, GHashTable* query, SoupClientContext* client_context, void* user_data )
{
	if( !user_data )
	{
		soup_message_set_status(message, SOUP_STATUS_INTERNAL_SERVER_ERROR);
		return;
	}
	
	WebRTCServer* server = (WebRTCServer*)user_data;
	
	// find if path is found
	HttpRoute* route = server->findHttpRoute(path);
	
	if( !route )
	{
		soup_message_set_status(message, SOUP_STATUS_NOT_FOUND);
		return;
	}
	
	// dispatch callback
	route->callback(soup_server, message, path, query, client_context, route->user_data);
}*/


// ProcessRequests
bool WebRTCServer::ProcessRequests( bool blocking )
{
	// https://stackoverflow.com/questions/23737750/glib-usage-without-mainloop
	// https://www.freedesktop.org/software/gstreamer-sdk/data/docs/2012.5/glib/glib-The-Main-Event-Loop.html#g-main-context-iteration
	// https://developer-old.gnome.org/programming-guidelines/stable/main-contexts.html.en
	g_main_context_iteration(NULL, blocking);
	return true;
}
