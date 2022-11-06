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

#ifndef __WEBRTC_SERVER_H__
#define __WEBRTC_SERVER_H__

#include <libsoup/soup.h>

#include <stdint.h>
#include <string>
#include <vector>


/**
 * Flags for route decorators or peer state.
 * @ingroup network
 */
enum WebRTCFlags
{
	// route/stream flags
	WEBRTC_PRIVATE = 0,		   // hidden (default)		
	WEBRTC_PUBLIC  = (1 << 1),  // allow discoverability
	WEBRTC_AUDIO   = (1 << 0),  // produces/accepts video
	WEBRTC_VIDEO   = (1 << 1),  // produces/accepts audio
	WEBRTC_SEND 	= (1 << 2),  // can send to the client
	WEBRTC_RECEIVE = (1 << 3),  // can recieve from the client

	// state flags
	WEBRTC_PEER_CONNECTING = (1 << 6),  // the peer is connecting
	WEBRTC_PEER_CONNECTED  = (1 << 7),  // the peer is ready
	WEBRTC_PEER_STREAMING  = (1 << 8),  // the peer is streaming
	WEBRTC_PEER_CLOSED     = (1 << 9),  // the peer has closed
};

//AddRoute("/my_video", my_function, this, WEBRTC_VIDEO|WEBRTC_SEND|WEBRTC_STREAM_PUBLIC);


class WebRTCServer;


/**
 * Remote peer that has connected.
 * @ingroup network
 */
struct WebRTCPeer
{
	SoupWebsocketConnection* connection;
	SoupClientContext* client_context;
	WebRTCServer* server;

	uint32_t ID;		 	// unique ID
	uint32_t flags;	 	// WebRTCFlags

	std::string path;   	// the route the peer is connected on
	std::string ip_address;	// IP address of the peer
	
	void* user_data;		// private data for the route handler
};


// TODO implement server caching by port

/**
 * GStreamer-based WebRTC http/websocket signalling server for establishing 
 * and negotiating connections with peers for bi-directional media streaming.   // full-duplex
 *
 * By default it serves simple HTML for viewing video streams in browsers, 
 * but for full hosting you'll want to run this alongside an actual webserver. 
 *
 * multi-stream :: multi-client
 * @ingroup network
 */
class WebRTCServer
{
public:
	// if port already in use, return an existing instance
	// html root path for serving files - if NULL, serve default   "disabled" or any string that doesn't resolve to files to disable alltogether
	// eventually change threaded default to true => ProcessMessages()
	// TODO added threaded (or flags?)
	// TODO implement server caching by port
	static WebRTCServer* Create( uint16_t port=8080 );	
	
	/**
	 * Release
	 * Server will be shut down when refcount reaches 0
	 */
	void Release();
	
	/**
	 * Function pointer to a callback for handling websocket requests.
	 * @see AddConnectionListener
	 */
	typedef void (*WebsocketListener)( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data );

	/**
	 * Function pointer to a callback for handling HTTP requests.
	 * @see https://developer-old.gnome.org/libsoup/stable/SoupServer.html#SoupServerCallback
	 */
	typedef SoupServerCallback HttpListener;

	/**
	 * Register a function to handle incoming http/www requests at the specified path. 
	 * If the path is "/" the callback will handle all routes.
	 * If the route for this path already exists, it will replace the existing route.
	 * To remove a route, set the callback to NULL.
	 */
	void AddRoute( const char* path, HttpListener callback, void* user_data=NULL, uint32_t flags=0 );

	/**
	 * Register a function to handle incoming Websocket requests at the specified path.
	 * If the path is "/" the callback will handle all routes.
	 * If the route for this path already exists, it will replace the existing route.
	 * To remove a route, set the callback to NULL.
	 */
	void AddRoute( const char* path, WebsocketListener callback, void* user_data=NULL, uint32_t flags=0 );

	/**
	 * Process incoming requests on the server.
	 * If set to blocking, the function can wait indefinitely for requests.
	 */
	bool ProcessRequests( bool blocking=false );
	
protected:

	WebRTCServer( uint16_t port );
	~WebRTCServer();
	
	bool init();

	static void onHttpRequest( SoupServer* soup_server, SoupMessage* message, const char* path, GHashTable* query, SoupClientContext* client_context, void* user_data );
	static void onHttpDefault( SoupServer* soup_server, SoupMessage* message, const char* path, GHashTable* query, SoupClientContext* client_context, void* user_data );
	
	static void onWebsocketOpened( SoupServer* server, SoupWebsocketConnection* connection, const char *path, SoupClientContext* client_context, void* user_data );
	static void onWebsocketMessage( SoupWebsocketConnection* connection, SoupWebsocketDataType data_type, GBytes* message, void* user_data );
	static void onWebsocketClosed( SoupWebsocketConnection* connection, void* user_data );
	
	struct HttpRoute
	{
		std::string path;
		HttpListener callback;
		void* user_data;
		uint32_t flags;
	};
	
	struct WebsocketRoute
	{
		std::string path;
		WebsocketListener callback;
		void* user_data;
		uint32_t flags;
		std::vector<WebRTCPeer*> peers;
	};

	HttpRoute* findHttpRoute( const char* path ) const;
	WebsocketRoute* findWebsocketRoute( const char* path ) const;
	
	void freeRoute( HttpRoute* route );
	void freeRoute( WebsocketRoute* route );
	
	std::vector<HttpRoute*> mHttpRoutes;
	std::vector<WebsocketRoute*> mWebsocketRoutes;
	
	uint16_t mPort;
	uint32_t mRefCount;
	uint32_t mPeerCount;
	
	SoupServer* mSoupServer;
};

#endif
