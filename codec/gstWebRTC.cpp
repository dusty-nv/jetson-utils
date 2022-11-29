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

#include "gstWebRTC.h"
#include "WebRTCServer.h"
#include "Networking.h"

#include "logging.h"

#include <json-glib/json-glib.h>
#include <regex>


// get_string_from_json_object
static gchar* get_string_from_json_object( JsonObject* object )
{
	JsonNode* root = json_node_init_object(json_node_alloc(), object);
	JsonGenerator* generator = json_generator_new();
	json_generator_set_root(generator, root);
	gchar* text = json_generator_to_data(generator, NULL);

	g_object_unref(generator);
	json_node_free(root);
	
	return text;
}


// onNegotiationNeeded
void gstWebRTC::onNegotiationNeeded( GstElement* webrtcbin, void* user_data )
{
	LogDebug(LOG_WEBRTC "GStreamer WebRTC -- onNegotiationNeeded()\n");
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;
	
	if( !peer )
		return;
	
	PeerContext* peer_context = (PeerContext*)peer->user_data;
	
	if( !peer_context )
		return;
	
	// setup offer created callback
	GstPromise* promise = gst_promise_new_with_change_func(onCreateOffer, peer, NULL);
	g_signal_emit_by_name(G_OBJECT(peer_context->webrtcbin), "create-offer", NULL, promise);
}


// onCreateOffer
void gstWebRTC::onCreateOffer( GstPromise* promise, void* user_data )
{
	LogDebug(LOG_WEBRTC "GStreamer WebRTC -- onCreateOffer()\n");
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;
	
	if( !peer )
		return;
	
	PeerContext* peer_context = (PeerContext*)peer->user_data;
	
	if( !peer_context )
		return;

	// send the SDP offer
	const GstStructure* reply = gst_promise_get_reply(promise);
	
	GstWebRTCSessionDescription* offer = NULL;
	gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
	gst_promise_unref(promise);

	GstPromise* local_desc_promise = gst_promise_new();
	g_signal_emit_by_name(peer_context->webrtcbin, "set-local-description", offer, local_desc_promise);
	gst_promise_interrupt(local_desc_promise);
	gst_promise_unref(local_desc_promise);

	gchar* sdp_string = gst_sdp_message_as_text(offer->sdp);
	LogVerbose(LOG_WEBRTC "negotiation offer created:\n%s\n", sdp_string);

	JsonObject* sdp_json = json_object_new();
	json_object_set_string_member(sdp_json, "type", "sdp");

	JsonObject* sdp_data_json = json_object_new ();
	json_object_set_string_member(sdp_data_json, "type", "offer");
	json_object_set_string_member(sdp_data_json, "sdp", sdp_string);
	json_object_set_object_member(sdp_json, "data", sdp_data_json);

	gchar* json_string = get_string_from_json_object(sdp_json);
	json_object_unref(sdp_json);

	LogVerbose(LOG_WEBRTC "sending offer for %s to %s (peer_id=%u): \n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, json_string);
	
	soup_websocket_connection_send_text(peer->connection, json_string);
	
	//g_free(json_string);
	g_free(sdp_string);
	gst_webrtc_session_description_free(offer);
}


// onIceCandidate
void gstWebRTC::onIceCandidate( GstElement* webrtcbin, uint32_t mline_index, char* candidate, void* user_data )
{
	LogDebug(LOG_WEBRTC "GStreamer WebRTC -- onICECandidate()\n");
	
	if( !user_data )
		return;
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;

	// send the ICE candidate
	JsonObject* ice_json = json_object_new();
	json_object_set_string_member(ice_json, "type", "ice");

	JsonObject* ice_data_json = json_object_new();
	json_object_set_int_member(ice_data_json, "sdpMLineIndex", mline_index);
	json_object_set_string_member(ice_data_json, "candidate", candidate);
	json_object_set_object_member(ice_json, "data", ice_data_json);

	gchar* json_string = get_string_from_json_object(ice_json);
	json_object_unref(ice_json);

	LogVerbose(LOG_WEBRTC "sending ICE candidate for %s to %s (peer_id=%u): \n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, json_string);

	soup_websocket_connection_send_text(peer->connection, json_string);
	
	g_free(json_string);
}


// Deal with MDNS candidates in ICE messages
// candidate:2612432513 1 udp 2113937151 968c736b-1028-4011-8977-df7a7c5eaea0.local 52811 typ host generation 0 ufrag thqf network-cost 999
// https://gitlab.freedesktop.org/gstreamer/gst-plugins-bad/-/issues/1139
// https://gitlab.freedesktop.org/gstreamer/gst-plugins-bad/-/issues/1344
static std::string resolveIceCandidate( const std::string& candidate )
{
	std::regex regex("(candidate:([0-9]*) ([0-9]*) (udp|UDP|tcp|TCP) ([0-9]*) (\\S*))");
	std::smatch matches;

	if( std::regex_search(candidate.begin(), candidate.end(), matches, regex) )
	{
		const uint32_t numMatches = matches.size();
		
		if( numMatches < 7 )
			printf("only %u matches\n", numMatches);
		
		const std::string match = matches.str(6);		// https://gitlab.freedesktop.org/gstreamer/gst-plugins-bad/-/issues/1344
		
		if( match.find(".local") != std::string::npos )
		{
			const std::string ipAddress = getHostByName(match.c_str());
			
			if( ipAddress.size() == 0 )
			{
				LogError(LOG_WEBRTC "couldn't resolve %s from SDP candidate string\n", match.c_str());
				return candidate;
			}
			
			LogVerbose(LOG_WEBRTC "resolved %s for %s in incoming ICE message\n", match.c_str(), ipAddress.c_str());

			const size_t org_position = candidate.find(match);
			
			if( org_position == std::string::npos )
			{
				LogError(LOG_WEBRTC "couldn't find '%s' in candidate string\n", match.c_str());
				return candidate;
			}
			
			std::string candidate_out = candidate;
			candidate_out.replace(org_position, match.size(), ipAddress);
			LogVerbose(LOG_WEBRTC "%s\n", candidate_out.c_str());
			return candidate_out;
		}
		
		/*for( uint32_t n=0; n < numMatches; n++ )
		{
			const std::string match = matches.str(n);
			std::cout << "  submatch " << matches.str(n) << "\n";
		}*/
	}
	else
		LogVerbose(LOG_WEBRTC "couldn't parse ICE candidate message with regex\n");

	return candidate;
}


// onWebsocketMessage
void gstWebRTC::onWebsocketMessage( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data )
{
	PeerContext* peer_context = (PeerContext*)peer->user_data;
	
	if( !peer_context )
		return;
	
	#define cleanup() { \
		if( json_parser != NULL ) \
			g_object_unref(G_OBJECT(json_parser)); \
		return; } \

	#define unknown_message() { \
		LogWarning(LOG_WEBRTC "gstEncoder -- unknown message, ignoring...\n%s\n", message); \
		cleanup(); }

	// parse JSON data string
	JsonParser* json_parser = json_parser_new();
	
	if( !json_parser_load_from_data(json_parser, message, -1, NULL) )
		unknown_message();

	JsonNode* root_json = json_parser_get_root(json_parser);
	
	if( !JSON_NODE_HOLDS_OBJECT(root_json) )
		unknown_message();

	JsonObject* root_json_object = json_node_get_object(root_json);

	// retrieve type string
	if( !json_object_has_member(root_json_object, "type") ) 
	{
		LogError(LOG_WEBRTC "received JSON message without 'type' field\n");
		cleanup();
	}
	
	const gchar* type_string = json_object_get_string_member(root_json_object, "type");

	// retrieve data object
	if( !json_object_has_member(root_json_object, "data") ) 
	{
		LogError(LOG_WEBRTC "received JSON message without 'data' field\n");
		cleanup();
	}
	
	JsonObject* data_json_object = json_object_get_object_member(root_json_object, "data");

	// handle message types
	if( g_strcmp0(type_string, "sdp") == 0 ) 
	{
		// validate SDP message
		if( !json_object_has_member(data_json_object, "type") ) 
		{
			LogError(LOG_WEBRTC "received SDP message without 'type' field\n");
			cleanup();
		}
		
		const gchar* sdp_type_string = json_object_get_string_member(data_json_object, "type");

		if( g_strcmp0(sdp_type_string, "answer") != 0 ) 
		{
			LogError(LOG_WEBRTC "expected SDP message type 'answer', got '%s'\n", sdp_type_string);
			cleanup();
		}

		if( !json_object_has_member(data_json_object, "sdp") )
		{
			LogError(LOG_WEBRTC "received SDP message without 'sdp' field\n");
			cleanup();
		}

		const gchar* sdp_string = json_object_get_string_member(data_json_object, "sdp");
		LogVerbose(LOG_WEBRTC "received SDP message for %s from %s (peer_id=%u)\n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, sdp_string);
		
		// parse SDP string
		GstSDPMessage* sdp = NULL;
		int ret = gst_sdp_message_new(&sdp);
		g_assert_cmphex(ret, ==, GST_SDP_OK);

		ret = gst_sdp_message_parse_buffer((guint8*)sdp_string, strlen(sdp_string), sdp);
		
		if( ret != GST_SDP_OK )
		{
			LogError(LOG_WEBRTC "failed to parse SDP string\n");
			cleanup();
		}

		// provide the SDP to webrtcbin
		GstWebRTCSessionDescription* answer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
		g_assert_nonnull(answer);

		GstPromise* promise = gst_promise_new();
		g_signal_emit_by_name(peer_context->webrtcbin, "set-remote-description", answer, promise);
		gst_promise_interrupt(promise);
		gst_promise_unref(promise);
		gst_webrtc_session_description_free(answer);
		
	}
	else if( g_strcmp0(type_string, "ice") == 0 )
	{
		// validate ICE message
		if( !json_object_has_member(data_json_object, "sdpMLineIndex") )
		{
			LogError(LOG_WEBRTC "received ICE message without 'sdpMLineIndex' field\n");
			cleanup();
		}
		
		const uint32_t mline_index = json_object_get_int_member(data_json_object, "sdpMLineIndex");

		// extract the ICE candidate
		if( !json_object_has_member(data_json_object, "candidate") ) 
		{
			LogError(LOG_WEBRTC "received ICE message without 'candidate' field\n");
			cleanup();
		}
		
		const gchar* candidate_string = json_object_get_string_member(data_json_object, "candidate");
		LogVerbose(LOG_WEBRTC "received ICE message on %s from %s (peer_id=%u) with mline index %u; candidate: \n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, mline_index, candidate_string);

		// resolve mDNS addresses
		const std::string candidate_resolved = resolveIceCandidate(candidate_string);	

		// provide the ICE candidate to webrtcbin
		g_signal_emit_by_name(peer_context->webrtcbin, "add-ice-candidate", mline_index, candidate_resolved.c_str());
	} 
	else
		unknown_message();

	cleanup();
}
