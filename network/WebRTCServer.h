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


// forward declarations
class WebRTCServer;
class Thread;


/**
 * Default HTTP/websocket port used by the WebRTC server.
 * @ingroup network
 */
#define WEBRTC_DEFAULT_PORT 41567

/**
 * Default STUN server used for WebRTC.
 * STUN servers are used during ICE/NAT and allow a local device to determine its public IP address.
 * @ingroup network
 */
#define WEBRTC_DEFAULT_STUN_SERVER "stun.l.google.com:19302"

/**
 * WebRTC logging prefix
 * @ingroup network
 */
#define LOG_WEBRTC "[webrtc] "


/**
 * Flags for route decorators or peer state.
 * @ingroup network
 */
enum WebRTCFlags
{
	// route/stream flags
	WEBRTC_PRIVATE = 0,				// hidden (default)		
	WEBRTC_PUBLIC  = (1 << 1),		// allow discoverability
	WEBRTC_AUDIO   = (1 << 0),		// produces/accepts video
	WEBRTC_VIDEO   = (1 << 1),  		// produces/accepts audio
	WEBRTC_SEND 	= (1 << 2),		// server sends to the client (outgoing)
	WEBRTC_RECEIVE = (1 << 3),		// server receives from the client (incoming)
	WEBRTC_MULTI_CLIENT = (1 << 4),	// can connect with multiple clients simultaneously
	
	// state flags
	WEBRTC_PEER_CONNECTING = (1 << 5),	// the peer is connecting
	WEBRTC_PEER_CONNECTED  = (1 << 6),	// the peer is ready
	WEBRTC_PEER_STREAMING  = (1 << 7),	// the peer is streaming
	WEBRTC_PEER_CLOSED     = (1 << 8),	// the peer has closed
};


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


/**
 * WebRTC signalling server for establishing and negotiating connections with peers 
 * for bi-directional media streaming. Users can bind routing functions that configure
 * their pipelines dynamically to handle incoming/outgoing streams.
 *
 * It uses websockets to handle the communication with clients for SDP offers, ICE messages, ect.
 * @see https://doc-kurento.readthedocs.io/en/stable/_static/client-javadoc/org/kurento/client/WebRtcEndpoint.html
 * 
 * By default it also serves simple HTML for viewing video streams in browsers, 
 * but for full hosting you'll want to run this alongside an actual webserver. 
 *
 * multi-stream :: multi-client :: full-duplex
 *
 * @ingroup network
 */
class WebRTCServer
{
public:
	/**
	 * Create a WebRTC server on this port.
	 * If this port is already in use, the existing server instance will be returned.
	 */
	static WebRTCServer* Create( uint16_t port=WEBRTC_DEFAULT_PORT, 
						    const char* stun_server=WEBRTC_DEFAULT_STUN_SERVER,
						    const char* ssl_cert=NULL, const char* ssl_key=NULL,
						    bool threaded=true );	
	
	/**
	 * Release a reference to the server instance.
	 * Server will be shut down when the reference count reaches zero.
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
	 * These are for serving webpages, and should be on different paths than the websockets.
	 * If the path is "/" the callback will handle all routes.
	 * If the route for this path already exists, it will replace the existing route.
	 * To remove a route, set the callback to NULL.
	 */
	void AddRoute( const char* path, HttpListener callback, void* user_data=NULL, uint32_t flags=0 );

	/**
	 * Register a function to handle incoming Websocket requests at the specified path.
	 *
	 * This path often represents a media stream that can be sent/recieved, 
	 * or a function for dynamically configuring streams when new clients connect.
	 *
	 * If the path is "/" the callback will handle all routes.
	 * If the route for this path already exists, it will replace the existing route.
	 * To remove a route, set the callback to NULL.
	 *
	 * The optional flags are WebRTCFlags OR'd together and are used for describing the 
	 * stream capabilities for purposes of discovery.
	 *
	 * @see WebRTCFlags
	 */
	void AddRoute( const char* path, WebsocketListener callback, void* user_data=NULL, uint32_t flags=0 );

	/**
	 * Get the STUN server being used.
	 * STUN servers are used during ICE/NAT and allow a local device to determine its public IP address.
	 * @see WEBRTC_DEFAULT_STUN_SERVER
	 */
	inline const char* GetSTUNServer() const		{ return mStunServer.c_str(); }
	
	/**
	 * Return true if the server is using HTTPS.
	 * For HTTPS to be enabled, SSL cert/key files must have been provided to WebRTCServer::Create().
	 */
	inline bool HasHTTPS() const					{ return mHasHTTPS; }
	
	/**
	 * Return true if the server is running in it's own thread.
	 * Otherwise, ProcessRequests() must be called periodically.
	 */
	inline bool IsThreaded() const				{ return (mThread != NULL); }
	
	/**
	 * Process incoming requests on the server.
	 * If set to blocking, the function can wait indefinitely for requests.
	 * This should only be called externally if the server was created with threaded=false
	 */
	bool ProcessRequests( bool blocking=false );
	
protected:

	WebRTCServer( uint16_t port, const char* stun_server, const char* ssl_cert_file, const char* ssl_key_file, bool threaded );
	~WebRTCServer();
	
	bool init();

	static void* runThread( void* user_data );
	
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
	
	std::string printRouteInfo( WebsocketRoute* route ) const;
	//std::string printRouteList();
	
	std::vector<HttpRoute*> mHttpRoutes;
	std::vector<WebsocketRoute*> mWebsocketRoutes;
	
	SoupServer* mSoupServer;
	std::string mStunServer;
	
	std::string mSSLCertFile;
	std::string mSSLKeyFile;
	
	bool mHasHTTPS;
	
	uint16_t mPort;
	uint32_t mRefCount;
	uint32_t mPeerCount;
	
	Thread* mThread;
};

#endif
