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
 
#include "RTSPServer.h"
#include "Networking.h"
#include "gstUtility.h"

#include "Thread.h"
#include "logging.h"

#include <gst/rtsp-server/rtsp-server.h>


// list of existing server instances
std::vector<RTSPServer*> gRTSPServers;

// map of media factories to pipeline elements
std::vector<std::pair<GstRTSPMediaFactory*, GstElement*>> gMediaFactoryMap;


// constructor
RTSPServer::RTSPServer( uint16_t port )
{	
	mPort = port;
	mRefCount = 1;
	mThread = new Thread();
	mRunning = false;
	mMainLoop = NULL;
	mServer = NULL;
}


// destructor
RTSPServer::~RTSPServer()
{
	if( mRunning )
	{
		g_main_loop_quit(mMainLoop);
		
		while(mRunning)
		{
			LogVerbose(LOG_RTSP "waiting for RTSP server to stop...\n");
			usleep(500 * 1000);
		}
		
		g_main_loop_unref(mMainLoop);
		mMainLoop = NULL;
	}
	
	if( mServer != NULL )
	{
		g_object_unref(mServer);
		mServer = NULL;
	}
			
	if( mThread != NULL )
	{
		delete mThread;
		mThread = NULL;
	}
}


// Release
void RTSPServer::Release()
{
	mRefCount--;
	
	if( mRefCount == 0 )
	{
		LogInfo(LOG_RTSP "RTSP server on port %hu is shutting down\n", mPort);
		
		for( size_t n=0; n < gRTSPServers.size(); n++ )
		{
			if( gRTSPServers[n] == this )
			{
				gRTSPServers.erase(gRTSPServers.begin() + n);
				break;
			}
		}

		delete this;
	}
}
		

// Create
RTSPServer* RTSPServer::Create( uint16_t port )
{
	// see if a server on this port already exists
	const uint32_t numServers = gRTSPServers.size();
	
	for( uint32_t n=0; n < numServers; n++ )
	{
		if( gRTSPServers[n]->mPort == port )
		{
			gRTSPServers[n]->mRefCount++;
			return gRTSPServers[n];
		}
	}

	// create a new server
	RTSPServer* server = new RTSPServer(port);

	if( !server )
	{
		LogError(LOG_RTSP "failed to create RTSP server on port %hu\n", port);
		return NULL;
	}
	
	// start the thread
	if( !server->mThread->Start(runThread, server) )
	{
		LogError(LOG_RTSP "failed to create thread for running RTSP server\n");
		return NULL;
	}

	// wait for initialization to complete
	for( uint32_t n=0; n < 15; n++ )
	{
		if( server->mRunning )
			break;
		
		LogVerbose(LOG_RTSP "waiting for RTSP server to start...\n");
		usleep(100 * 1000);
	}
	
	if( !server->mRunning )
	{
		LogError(LOG_RTSP "failed to start RTSP server on port %hu\n", server->mPort);
		return NULL;
	}
		
	gRTSPServers.push_back(server);
	return server;
}


// init
bool RTSPServer::init()
{
	// initialize GStreamer libraries
	if( !gstreamerInit() )
	{
		LogError(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return false;
	}
	
	// make a main loop for the default context
	mMainLoop = g_main_loop_new(NULL, false);
	
	if( !mMainLoop )
	{
		LogError(LOG_RTSP "failed to create GMainLoop instance\n");
		return false;
	}
	
	// make a server instance
	mServer = gst_rtsp_server_new();
	
	if( !mServer )
	{
		LogError(LOG_RTSP "failed to create GstRTSPServer instance\n");
		return false;
	}
	
	// set the port
	char port_str[16];
	sprintf(port_str, "%hu", mPort);
	gst_rtsp_server_set_service(mServer, port_str);
	
	// attach the server to the default maincontext
     if( gst_rtsp_server_attach(mServer, NULL) == 0 )
	{
		LogError(LOG_RTSP "failed to attach server to port %hu\n", mPort);
		return false;
	}
	
	LogSuccess(LOG_RTSP "RTSP server started @ rtsp://%s:%hu\n", getHostname().c_str(), mPort);
	return true;
}


// runThread
void* RTSPServer::runThread( void* user_data )
{
	RTSPServer* server = (RTSPServer*)user_data;
	
	if( !server )
		return 0;
	
	// initialize server resources
	if( !server->init() )
	{
		LogError(LOG_RTSP "failed to create RTSP server on port %hu\n", server->mPort);
		return 0;
	}
	
	// run the server's main loop
	server->mRunning = true;
	g_main_loop_run(server->mMainLoop);
	server->mRunning = false;
	
	LogVerbose(LOG_RTSP "RTSP server thread stopped\n");
	return 0;
}


// custom implementation of GstRTSPMediaFactory::create_element()
static GstElement* gst_rtsp_media_factory_custom_element( GstRTSPMediaFactory* factory, const GstRTSPUrl* url )
{
	const uint32_t numFactories = gMediaFactoryMap.size();
	
	for( uint32_t n=0; n < numFactories; n++ )
	{
		if( factory == gMediaFactoryMap[n].first )
			return gMediaFactoryMap[n].second;
	}
	
	LogError(LOG_RTSP "failed to lookup media factory pipeline element\n");
	return NULL;
}
    
    
// apply some additional settings on each GstRTSPMedia object
static void gst_rtsp_media_factory_custom_configure( GstRTSPMediaFactory* factory, GstRTSPMedia* media, gpointer user_data )
{
	gst_rtsp_media_set_reusable(media, true);
	gst_rtsp_media_prepare(media, NULL);
	gst_rtsp_media_set_pipeline_state(media, GST_STATE_PLAYING);
}

    
// AddRoute
bool RTSPServer::AddRoute( const char* path, GstElement* pipeline )
{
	// get the mount points for the server
	GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(mServer);
	
	if( !mounts )
	{
		LogError(LOG_RTSP "AddRoute() -- failed to get mount points from RTSP server\n");
		return false;
	}
	
	// redirect the media factory create_element() to our function
	GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
	GstRTSPMediaFactoryClass* factoryFunctions = GST_RTSP_MEDIA_FACTORY_GET_CLASS(factory);
	
	factoryFunctions->create_element = gst_rtsp_media_factory_custom_element;
	gMediaFactoryMap.push_back( std::pair<GstRTSPMediaFactory*, GstElement*>(factory, pipeline) );
	
	// setup media streaming options
	gst_rtsp_media_factory_set_latency(factory, 0);
	gst_rtsp_media_factory_set_buffer_size(factory, 0);
	gst_rtsp_media_factory_set_shared(factory, true);
	gst_rtsp_media_factory_set_eos_shutdown(factory, false);
	gst_rtsp_media_factory_set_stop_on_disconnect(factory, false);
	gst_rtsp_media_factory_set_suspend_mode(factory, GST_RTSP_SUSPEND_MODE_NONE);
	gst_rtsp_media_factory_set_transport_mode(factory, GST_RTSP_TRANSPORT_MODE_PLAY);
	
#if GST_CHECK_VERSION(1,16,0)
	gst_rtsp_media_factory_set_do_retransmission(factory, false);
#endif

	g_signal_connect(factory, "media-configure", (GCallback)gst_rtsp_media_factory_custom_configure, NULL);
	 
	// attach the factory to the url
	gst_rtsp_mount_points_add_factory(mounts, path, factory);
	g_object_unref(mounts);
	
	LogVerbose(LOG_RTSP "RTSP route added %s @ rtsp://%s:%hu\n", path, getHostname().c_str(), mPort);
	return true;
}


// AddRoute
bool RTSPServer::AddRoute( const char* path, const char* pipeline_str )
{
	GError* err = NULL;
	GstElement* pipeline = gst_parse_launch_full(pipeline_str, NULL, GST_PARSE_FLAG_PLACE_IN_BIN, &err);

	if( err != NULL )
	{
		LogError(LOG_GSTREAMER "failed to create pipeline:\n");
		LogError(LOG_GSTREAMER "   %s\n", pipeline_str);
		LogError(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}
	
	return AddRoute(path, pipeline);
}