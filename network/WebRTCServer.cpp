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
	   console.log(\"Video server URL: \" + wsUrl); \n \
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


// list of existing servers
std::vector<WebRTCServer*> gServers;


// constructor
WebRTCServer::WebRTCServer( uint16_t port )
{	
	mPort = port;
	mRefCount = 1;
	mPeerCount = 0;
	mSoupServer = NULL;
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
		LogInfo(LOG_WEBRTC "server on port %hu is shutting down\n", mPort);
		vector_remove_element(gServers, this);
		delete this;
	}
}
		

// Create
WebRTCServer* WebRTCServer::Create( uint16_t port )
{
	// see if a server on this port already exists
	const uint32_t numServers = gServers.size();
	
	for( uint32_t n=0; n < numServers; n++ )
	{
		if( gServers[n]->mPort == port )
			return gServers[n];
	}
	
	// create a new server
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
	//soup_server_add_websocket_handler(mSoupServer, "/", NULL, NULL, onWebsocketOpened, this, NULL);
	
	AddRoute("/", onHttpDefault, this);  // serve the server-default HTML pages
	
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
		if( server->mHttpRoutes.size() == 1 && server->mHttpRoutes[0]->path == "/" )
			route = server->mHttpRoutes[0];
	}

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
	
#if 1
	// find if path is found
	WebsocketRoute* websocketRoute = server->findWebsocketRoute(path);

	if( !websocketRoute )
	{
		if( strcmp(path, "/") == 0 || strcmp(path, "index.html") == 0 )
			html = html_viewer;
		else
			html = html_inactive;
	}
	else
	{
		html = html_viewer;
	}
#else
	if( server->mWebsocketRoutes.size() > 0 )
		html = html_viewer;
#endif

	if( !html )
	{
		LogVerbose(LOG_WEBRTC "HTTP %s %s '%s' -- not found 404\n", soup_client_context_get_host(client_context), message->method, path);
		soup_message_set_status(message, SOUP_STATUS_NOT_FOUND);
		return;
	}
		
	SoupBuffer* soup_buffer = soup_buffer_new(SOUP_MEMORY_STATIC, html, strlen(html)); // TODO watch SOUP_MEMORY_STATIC when we start making other strings

	soup_message_headers_set_content_type(message->response_headers, "text/html", NULL);
	soup_message_body_append_buffer(message->response_body, soup_buffer);
	soup_buffer_free(soup_buffer);

	soup_message_set_status(message, SOUP_STATUS_OK);
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
