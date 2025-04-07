/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "gstUtility.h"
#include "filesystem.h"

#include "NvInfer.h"
#include "logging.h"

#include <stdint.h>
#include <stdio.h>
#include <strings.h>
#include <algorithm>


//---------------------------------------------------------------------------------------------
imageFormat gst_parse_format( GstStructure* caps )
{
	const char* format = gst_structure_get_string(caps, "format");
	
	if( !format )
		return IMAGE_UNKNOWN;
	
	if( strcasecmp(format, "rgb") == 0 )
		return IMAGE_RGB8;
	else if( strcasecmp(format, "yuy2") == 0 )
		return IMAGE_YUY2;
	else if( strcasecmp(format, "i420") == 0 )
		return IMAGE_I420;
	else if( strcasecmp(format, "nv12") == 0 )
		return IMAGE_NV12;
	else if( strcasecmp(format, "yv12") == 0 )
		return IMAGE_YV12;
	else if( strcasecmp(format, "yuyv") == 0 )
		return IMAGE_YUYV;
	else if( strcasecmp(format, "yvyu") == 0 )
		return IMAGE_YVYU;
	else if( strcasecmp(format, "uyvy") == 0 )
		return IMAGE_UYVY;
	else if( strcasecmp(format, "bggr") == 0 )
		return IMAGE_BAYER_BGGR;
	else if( strcasecmp(format, "gbrg") == 0 )
		return IMAGE_BAYER_GBRG;
	else if( strcasecmp(format, "grgb") == 0 )
		return IMAGE_BAYER_GRBG;
	else if( strcasecmp(format, "rggb") == 0 )
		return IMAGE_BAYER_RGGB;
	else if( strcasecmp(format, "gray8") == 0 )
		return IMAGE_GRAY8;
	
	return IMAGE_UNKNOWN;
}

const char* gst_format_to_string( imageFormat format )
{
	switch(format)
	{
		case IMAGE_RGB8:	return "RGB";
		case IMAGE_YUY2:	return "YUY2";
		case IMAGE_I420:	return "I420";
		case IMAGE_NV12:	return "NV12";
		case IMAGE_YV12:	return "YV12";
		case IMAGE_YVYU:	return "YVYU";
		case IMAGE_UYVY:	return "UYVY";
		case IMAGE_BAYER_BGGR:	return "bggr";
		case IMAGE_BAYER_GBRG:	return "gbrg";
		case IMAGE_BAYER_GRBG:	return "grbg";
		case IMAGE_BAYER_RGGB:	return "rggb";
		case IMAGE_GRAY8:	return "gray8";
	}
	
	return " ";
}

videoOptions::Codec gst_parse_codec( GstStructure* caps )
{
	const char* codec = gst_structure_get_name(caps);
	
	if( !codec )
		return videoOptions::CODEC_UNKNOWN;
	
	if( strcasecmp(codec, "video/x-raw") == 0 || strcasecmp(codec, "video/x-bayer") == 0 )
		return videoOptions::CODEC_RAW;
	else if( strcasecmp(codec, "video/x-h264") == 0 )
		return videoOptions::CODEC_H264;
	else if( strcasecmp(codec, "video/x-h265") == 0 )
		return videoOptions::CODEC_H265;
	else if( strcasecmp(codec, "video/x-vp8") == 0 )
		return videoOptions::CODEC_VP8;
	else if( strcasecmp(codec, "video/x-vp9") == 0 )
		return videoOptions::CODEC_VP9;
	else if( strcasecmp(codec, "image/jpeg") == 0 )
		return videoOptions::CODEC_MJPEG;
	else if( strcasecmp(codec, "video/mpeg") == 0 )
	{
		int mpegVersion = 0;
	
		if( !gst_structure_get_int(caps, "mpegversion", &mpegVersion) )
		{
			LogError(LOG_GSTREAMER "MPEG codec, but failed to get MPEG version from caps\n");
			return videoOptions::CODEC_UNKNOWN;
		}
		
		if( mpegVersion == 2 )
			return videoOptions::CODEC_MPEG2;
		else if( mpegVersion == 4 )
			return videoOptions::CODEC_MPEG4;
		else
		{
			LogError(LOG_GSTREAMER "invalid MPEG codec version:  %i (MPEG-2 and MPEG-4 are supported)\n", mpegVersion);
			return videoOptions::CODEC_UNKNOWN;
		}
	}
	
	LogError(LOG_GSTREAMER "unrecognized codec - %s\n", codec);
	return videoOptions::CODEC_UNKNOWN;
}

const char* gst_codec_to_string( videoOptions::Codec codec )
{
	switch(codec)
	{
		case videoOptions::CODEC_RAW: 	return "video/x-raw";
		case videoOptions::CODEC_H264:	return "video/x-h264";
		case videoOptions::CODEC_H265:	return "video/x-h265";
		case videoOptions::CODEC_VP8:	return "video/x-vp8";
		case videoOptions::CODEC_VP9:	return "video/x-vp9";
		case videoOptions::CODEC_MJPEG:	return "image/jpeg";
		case videoOptions::CODEC_MPEG2:	return "video/mpeg, mpegversion=(int)2";
		case videoOptions::CODEC_MPEG4:	return "video/mpeg, mpegversion=(int)4";
	}
	
	return " ";
}


//---------------------------------------------------------------------------------------------
inline const char* gst_debug_level_str( GstDebugLevel level )
{
	switch (level)
	{
		case GST_LEVEL_NONE:	return "GST_LEVEL_NONE   ";
		case GST_LEVEL_ERROR:	return "GST_LEVEL_ERROR  ";
		case GST_LEVEL_WARNING:	return "GST_LEVEL_WARNING";
		case GST_LEVEL_INFO:	return "GST_LEVEL_INFO   ";
		case GST_LEVEL_DEBUG:	return "GST_LEVEL_DEBUG  ";
		case GST_LEVEL_LOG:		return "GST_LEVEL_LOG    ";
		case GST_LEVEL_FIXME:	return "GST_LEVEL_FIXME  ";
#ifdef GST_LEVEL_TRACE
		case GST_LEVEL_TRACE:	return "GST_LEVEL_TRACE  ";
#endif
		case GST_LEVEL_MEMDUMP:	return "GST_LEVEL_MEMDUMP";
    		default:				return "<unknown>        ";
    }
}

#define SEP "              "

void rilog_debug_function(GstDebugCategory* category, GstDebugLevel level,
                          const gchar* file, const char* function,
                          gint line, GObject* object, GstDebugMessage* message,
                          gpointer data)
{
	if( level > GST_LEVEL_WARNING /*GST_LEVEL_INFO*/ )
		return;

	//gchar* name = NULL;
	//if( object != NULL )
	//	g_object_get(object, "name", &name, NULL);

	const char* typeName  = " ";
	const char* className = " ";

	if( object != NULL )
	{
		typeName  = G_OBJECT_TYPE_NAME(object);
		className = G_OBJECT_CLASS_NAME(object);
	}

	LogVerbose(LOG_GSTREAMER "%s %s %s\n" SEP "%s:%i  %s\n" SEP "%s\n", 
		  	 gst_debug_level_str(level), typeName,
		  	 gst_debug_category_get_name(category), file, line, function, 
            	 gst_debug_message_get(message));

}


// gstreamerInit
bool gstreamerInit()
{
	static bool gstreamer_initialized = false;

	if( gstreamer_initialized )
		return true;

	int argc = 0;
	//char* argv[] = { "none" };

	if( !gst_init_check(&argc, NULL, NULL) )
	{
		LogError(LOG_GSTREAMER "failed to initialize gstreamer library with gst_init()\n");
		return false;
	}

	gstreamer_initialized = true;

	uint32_t ver[] = { 0, 0, 0, 0 };
	gst_version( &ver[0], &ver[1], &ver[2], &ver[3] );

	LogInfo(LOG_GSTREAMER "initialized gstreamer, version %u.%u.%u.%u\n", ver[0], ver[1], ver[2], ver[3]);


	// debugging
	gst_debug_remove_log_function(gst_debug_log_default);
	
	if( true )
	{
		gst_debug_add_log_function(rilog_debug_function, NULL, NULL);

		gst_debug_set_active(true);
		gst_debug_set_colored(false);
	}
	
	return true;
}

//---------------------------------------------------------------------------------------------
static void gst_print_one_tag(const GstTagList * list, const gchar * tag, gpointer user_data)
{
  int i, num;

  num = gst_tag_list_get_tag_size (list, tag);
  for (i = 0; i < num; ++i) {
    const GValue *val;

    /* Note: when looking for specific tags, use the gst_tag_list_get_xyz() API,
     * we only use the GValue approach here because it is more generic */
    val = gst_tag_list_get_value_index (list, tag, i);
    if (G_VALUE_HOLDS_STRING (val)) {
      LogVerbose("\t%20s : %s\n", tag, g_value_get_string (val));
    } else if (G_VALUE_HOLDS_UINT (val)) {
      LogVerbose("\t%20s : %u\n", tag, g_value_get_uint (val));
    } else if (G_VALUE_HOLDS_DOUBLE (val)) {
      LogVerbose("\t%20s : %g\n", tag, g_value_get_double (val));
    } else if (G_VALUE_HOLDS_BOOLEAN (val)) {
      LogVerbose("\t%20s : %s\n", tag,
          (g_value_get_boolean (val)) ? "true" : "false");
    } else if (GST_VALUE_HOLDS_BUFFER (val)) {
      //GstBuffer *buf = gst_value_get_buffer (val);
      //guint buffer_size = GST_BUFFER_SIZE(buf);

      LogVerbose("\t%20s : buffer of size %u\n", tag, /*buffer_size*/0);
    } /*else if (GST_VALUE_HOLDS_DATE_TIME (val)) {
      GstDateTime *dt = (GstDateTime*)g_value_get_boxed (val);
      gchar *dt_str = gst_date_time_to_iso8601_string (dt);

      printf("\t%20s : %s\n", tag, dt_str);
      g_free (dt_str);
    }*/ else {
      LogVerbose("\t%20s : tag of type '%s'\n", tag, G_VALUE_TYPE_NAME (val));
    }
  }
}

static const char* gst_stream_status_string( GstStreamStatusType status )
{
	switch(status)
	{
		case GST_STREAM_STATUS_TYPE_CREATE:	return "CREATE";
		case GST_STREAM_STATUS_TYPE_ENTER:		return "ENTER";
		case GST_STREAM_STATUS_TYPE_LEAVE:		return "LEAVE";
		case GST_STREAM_STATUS_TYPE_DESTROY:	return "DESTROY";
		case GST_STREAM_STATUS_TYPE_START:		return "START";
		case GST_STREAM_STATUS_TYPE_PAUSE:		return "PAUSE";
		case GST_STREAM_STATUS_TYPE_STOP:		return "STOP";
		default:							return "UNKNOWN";
	}
}

// gst_message_print
gboolean gst_message_print(GstBus* bus, GstMessage* message, gpointer user_data)
{
	switch (GST_MESSAGE_TYPE (message)) 
	{
		case GST_MESSAGE_ERROR: 
		{
			GError *err = NULL;
			gchar *dbg_info = NULL;
 
			gst_message_parse_error (message, &err, &dbg_info);
			LogVerbose(LOG_GSTREAMER "gstreamer %s ERROR %s\n", GST_OBJECT_NAME (message->src), err->message);
        		LogVerbose(LOG_GSTREAMER "gstreamer Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
        
			g_error_free(err);
        		g_free(dbg_info);
			//g_main_loop_quit (app->loop);
        		break;
		}
		case GST_MESSAGE_EOS:
		{
			LogVerbose(LOG_GSTREAMER "gstreamer %s recieved EOS signal...\n", GST_OBJECT_NAME(message->src));
			//g_main_loop_quit (app->loop);		// TODO trigger plugin Close() upon error
			break;
		}
		case GST_MESSAGE_STATE_CHANGED:
		{
			GstState old_state, new_state;
    
			gst_message_parse_state_changed(message, &old_state, &new_state, NULL);
			
			LogVerbose(LOG_GSTREAMER "gstreamer changed state from %s to %s ==> %s\n",
							gst_element_state_get_name(old_state),
							gst_element_state_get_name(new_state),
						     GST_OBJECT_NAME(message->src));
			break;
		}
		case GST_MESSAGE_STREAM_STATUS:
		{
			GstStreamStatusType streamStatus;
			gst_message_parse_stream_status(message, &streamStatus, NULL);
			
			LogVerbose(LOG_GSTREAMER "gstreamer stream status %s ==> %s\n",
							gst_stream_status_string(streamStatus), 
							GST_OBJECT_NAME(message->src));
			break;
		}
		case GST_MESSAGE_TAG: 
		{
			GstTagList *tags = NULL;
			gst_message_parse_tag(message, &tags);
			gchar* txt = gst_tag_list_to_string(tags);

			if( txt != NULL )
			{
				LogVerbose(LOG_GSTREAMER "gstreamer %s %s\n", GST_OBJECT_NAME(message->src), txt);		
				g_free(txt);	
			}
		
			//gst_tag_list_foreach(tags, gst_print_one_tag, NULL);

			if( tags != NULL )			
				gst_tag_list_free(tags);
			
			break;
		}
		default:
		{
			LogVerbose(LOG_GSTREAMER "gstreamer message %s ==> %s\n", gst_message_type_get_name(GST_MESSAGE_TYPE(message)), GST_OBJECT_NAME(message->src));
			break;
		}
	}

	return TRUE;
}


// gst_build_filesink
bool gst_build_filesink( const URI& uri, videoOptions::Codec codec, std::ostringstream& pipeline )
{
	if( uri.path.length() <= 0 || uri.protocol != "file" )
	{
		LogError(LOG_GSTREAMER "invalid file path -- unable to build filesink pipeline\n");
		return false;
	}
	
	#define ADD_CODEC_PARSER() \
		if( codec == videoOptions::CODEC_H264 ) \
			pipeline << "h264parse ! "; \
		else if( codec == videoOptions::CODEC_H265 ) \
			pipeline << "h265parse ! ";
		
	if( uri.extension == "mkv" )
	{
		ADD_CODEC_PARSER();
		pipeline << "matroskamux ! ";
	}
	else if( uri.extension == "flv" )
	{
		ADD_CODEC_PARSER();
		pipeline << "flvmux ! ";
	}
	else if( uri.extension == "avi" )
	{
		if( codec == videoOptions::CODEC_H265 || codec == videoOptions::CODEC_VP9 )
		{
			LogError(LOG_GSTREAMER "AVI format doesn't support codec %s\n", videoOptions::CodecToStr(codec));
			LogError(LOG_GSTREAMER "supported AVI codecs are:\n");
			LogError(LOG_GSTREAMER "   * h264\n");
			LogError(LOG_GSTREAMER "   * vp8\n");
			LogError(LOG_GSTREAMER "   * mjpeg\n");

			return false;
		}

		pipeline << "avimux ! ";
	}
	else if( uri.extension == "mp4" || uri.extension == "qt" )
	{
		ADD_CODEC_PARSER();
		pipeline << "qtmux ! ";
	}
	else if( uri.extension != "h264" && uri.extension != "h265" )
	{
		printf(LOG_GSTREAMER "unsupported video file extension (%s)\n", uri.extension.c_str());
		printf(LOG_GSTREAMER "supported video extensions are:\n");
		printf(LOG_GSTREAMER "   * mkv\n");
		printf(LOG_GSTREAMER "   * mp4, qt\n");
		printf(LOG_GSTREAMER "   * flv\n");
		printf(LOG_GSTREAMER "   * avi\n");
		printf(LOG_GSTREAMER "   * h264, h265\n");

		return false;
	}

	pipeline << "filesink location=" << uri.location << " ";
	return true;
}


// gst_select_decoder
const char* gst_select_decoder( videoOptions::Codec codec, videoOptions::CodecType& type )
{
#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
	if( type == videoOptions::CODEC_OMX )  // JetPack 5 doesn't have OMX
		type = gst_default_codec();
#endif
#elif defined(__x86_64__) || defined(__amd64__)
	if( type == videoOptions::CODEC_OMX || type == videoOptions::CODEC_V4L2 )
		type = gst_default_codec();
#endif

	if( type == videoOptions::CODEC_NVENC || type == videoOptions::CODEC_NVDEC )
		type = gst_default_codec();
	
	if( codec == videoOptions::CODEC_MJPEG )
		type = videoOptions::CODEC_CPU;
	
	if( codec == videoOptions::CODEC_RAW )
		type = videoOptions::CODEC_CPU;
	
	if( type == videoOptions::CODEC_CPU )
	{
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "avdec_h264";
			case videoOptions::CODEC_H265:   return "avdec_h265";
			case videoOptions::CODEC_VP8:	   return "vp8dec";
			case videoOptions::CODEC_VP9:    return "vp9dec";
			case videoOptions::CODEC_MPEG2:  return "avdec_mpeg2video";
			case videoOptions::CODEC_MPEG4:  return "avdec_mpeg4";
			case videoOptions::CODEC_MJPEG:  return "jpegdec";
		}
	}
	else if( type == videoOptions::CODEC_OMX )
	{
	#if GST_CHECK_VERSION(1,0,0)
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "omxh264dec";
			case videoOptions::CODEC_H265:   return "omxh265dec";
			case videoOptions::CODEC_VP8:	   return "omxvp8dec";
			case videoOptions::CODEC_VP9:    return "omxvp9dec";
			case videoOptions::CODEC_MPEG2:  return "omxmpeg2videodec";
			case videoOptions::CODEC_MPEG4:  return "omxmpeg4videodec";
			case videoOptions::CODEC_MJPEG:  return "nvjpegdec";
		}
	#else
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "nv_omx_h264dec";
			case videoOptions::CODEC_H265:   return "nv_omx_h265dec";
			case videoOptions::CODEC_VP8:	   return "nv_omx_vp8dec";
			case videoOptions::CODEC_VP9:    return "nv_omx_vp9dec";
			case videoOptions::CODEC_MPEG2:  return "nx_omx_mpeg2videodec";
			case videoOptions::CODEC_MPEG4:  return "nx_omx_mpeg4videodec";
			case videoOptions::CODEC_MJPEG:  return "nvjpegdec";
		}
	#endif
	}
	else if( type == videoOptions::CODEC_V4L2 )
	{
		if( codec == videoOptions::CODEC_MJPEG )
			return "nvjpegdec";
		
		return "nvv4l2decoder";
	}
	
	return NULL;
}


// check for hardware-accelerated encoder support
static bool gst_query_hw_encoder()
{
#if defined(__aarch64__)
	std::string board = readFile("/proc/device-tree/model");
	
	if( board.length() == 0 )
		board = readFile("/tmp/nv_jetson_model");  // this is where it gets mounted in the container
	
	if( board.length() == 0 )
		return false;
	
	LogVerbose(LOG_GSTREAMER "gstEncoder -- detected board '%s'\n", board.c_str());
	
	// convert to lowercase for robustness
	std::transform(board.begin(), board.end(), board.begin(), [](unsigned char c){ return std::tolower(c); });
	
	// look for specific boards that don't have encoder hw
	if( board.find("orin nano") != std::string::npos )
		return false;
	
	return true;
#else
	// TODO check for NVENC/NVDEC
	return false;
#endif
}


// gst_select_encoder
const char* gst_select_encoder( videoOptions::Codec codec, videoOptions::CodecType& type )
{
#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
	if( type == videoOptions::CODEC_OMX )
	{
		// JetPack 5 doesn't have OMX
		type = gst_default_codec();
	}
	else if( type == videoOptions::CODEC_V4L2 )
	{
		const bool has_hw_encoder = gst_query_hw_encoder();
		
		if( !has_hw_encoder )
		{
			LogWarning(LOG_GSTREAMER "gstEncoder -- hardware encoder not detected, reverting to CPU encoder\n");
			type = videoOptions::CODEC_CPU;
		}
	}
#endif
#elif defined(__x86_64__) || defined(__amd64__)
	if( type == videoOptions::CODEC_OMX || type == videoOptions::CODEC_V4L2 )
		type = gst_default_codec();
#endif

	if( type == videoOptions::CODEC_NVENC || type == videoOptions::CODEC_NVDEC )
		type = gst_default_codec();  // TODO NVENC/NVDEC support
	
	if( codec == videoOptions::CODEC_RAW )
		type = videoOptions::CODEC_CPU;
	
	if( type == videoOptions::CODEC_CPU )
	{
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "x264enc";
			case videoOptions::CODEC_H265:   return "x265enc";
			case videoOptions::CODEC_VP8:	   return "vp8enc";
			case videoOptions::CODEC_VP9:    return "vp9enc";
			case videoOptions::CODEC_MJPEG:  return "jpegenc";
		}
	}
	else if( type == videoOptions::CODEC_OMX )
	{
	#if GST_CHECK_VERSION(1,0,0)
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "omxh264enc";
			case videoOptions::CODEC_H265:   return "omxh265enc";
			case videoOptions::CODEC_VP8:	   return "omxvp8enc";
			case videoOptions::CODEC_VP9:    return "omxvp9enc";
			case videoOptions::CODEC_MJPEG:  return "nvjpegenc";
		}
	#else
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "nv_omx_h264enc";
			case videoOptions::CODEC_H265:   return "nv_omx_h265enc";
			case videoOptions::CODEC_VP8:	   return "nv_omx_vp8enc";
			case videoOptions::CODEC_VP9:    return "nv_omx_vp9enc";
			case videoOptions::CODEC_MJPEG:  return "nvjpegenc";
		}
	#endif
	}
	else if( type == videoOptions::CODEC_V4L2 )
	{
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "nvv4l2h264enc";
			case videoOptions::CODEC_H265:   return "nvv4l2h265enc";
			case videoOptions::CODEC_VP8:	   return "nvv4l2vp8enc";
			case videoOptions::CODEC_VP9:    return "nvv4l2vp9enc";
			case videoOptions::CODEC_MJPEG:  return "nvjpegenc";
		}
	}
	
	return NULL;
}


// gst_default_codec_type
videoOptions::CodecType gst_default_codec()
{
#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
	return videoOptions::CODEC_V4L2;	// JetPack 5
#else
	return videoOptions::CODEC_OMX;	// JetPack 4
#endif
#elif defined(__x86_64__) || defined(__amd64__)
	return videoOptions::CODEC_CPU;	// x86
#endif
}

