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

#ifndef __GSTREAMER_WEBRTC_H__
#define __GSTREAMER_WEBRTC_H__

#include "WebRTCServer.h"
#include "gstUtility.h"

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>


/**
 * Static class for common WebRTC utility functions used with GStreamer.
 * This gets used internally by gstEncoder/gstDecoder for handling WebRTC streams.
 * @ingroup codec
 */
class gstWebRTC
{
public:
	/**
	 * GStreamer-specific context for each WebRTCPeer
	 */
	struct PeerContext
	{
		PeerContext()	{ webrtcbin = NULL; queue = NULL; }
		
		GstElement* webrtcbin;	// used by gstEncoder + gstDecoder
		GstElement* queue;		// used by gstEncoder only
	};

	/**
	 * Callback for handling webrtcbin "on-negotation-needed" signal.
	 * It's expected that user_data is set to a WebRTCPeer instance.
	 */
	static void onNegotiationNeeded( GstElement* webrtcbin, void* user_data );
	
	/**
	 * Callback for handling webrtcbin "create-offer" signal.
	 * This sends an SDP offer to the client.
	 */
	static void onCreateOffer( GstPromise* promise, void* user_data );
	
	/**
	 * Callback for handling webrtcbin "on-ice-candidate" signal.
	 * This send an ICE candidate to the client.
	 */
	static void onIceCandidate( GstElement* webrtcbin, uint32_t mline_index, char* candidate, void* user_data );

	/**
	 * Handle incoming websocket messages from the client.
	 * This only handles SDP/ICE messages - it's expected that the caller will 
	 * handle new peer connecting/closing messages.
	 */
	static void onWebsocketMessage( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data );	
};


#endif