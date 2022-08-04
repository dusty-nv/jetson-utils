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

#ifndef __GSTREAMER_UTILITY_H__
#define __GSTREAMER_UTILITY_H__

#include <gst/gst.h>

#include "videoOptions.h"
#include "NvInfer.h"


/**
 * LOG_GSTREAMER logging prefix
 * @ingroup codec
 */
#define LOG_GSTREAMER "[gstreamer] "


/**
 * gstreamerInit
 * @internal
 * @ingroup codec
 */
bool gstreamerInit();

/**
 * gst_message_print
 * @internal
 * @ingroup codec
 */
gboolean gst_message_print(_GstBus* bus, _GstMessage* message, void* user_data);

/**
 * gst_parse_codec
 * @internal
 * @ingroup codec
 */
videoOptions::Codec gst_parse_codec( GstStructure* caps );

/**
 * gst_parse_format
 * @internal
 * @ingroup codec
 */
imageFormat gst_parse_format( GstStructure* caps );

/**
 * gst_codec_to_string
 * @internal
 * @ingroup codec
 */
const char* gst_codec_to_string( videoOptions::Codec codec );

/**
 * gst_format_to_string
 * @internal
 * @ingroup codec
 */
const char* gst_format_to_string( imageFormat format );


#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 4

/**
 * Use nvv4l2 codecs for JetPack 5 and newer
 * @internal
 * @ingroup codec
 */
#define GST_CODECS_V4L2

// Decoders for JetPack >= 5 and GStreamer >= 1.0
#define GST_DECODER_H264  "nvv4l2decoder"
#define GST_DECODER_H265  "nvv4l2decoder"
#define GST_DECODER_VP8   "nvv4l2decoder"
#define GST_DECODER_VP9   "nvv4l2decoder"
#define GST_DECODER_MPEG2 "nvv4l2decoder"
#define GST_DECODER_MPEG4 "nvv4l2decoder"
#define GST_DECODER_MJPEG "nvjpegdec"

// Encoders for JetPack >= 5 and GStreamer >= 1.0
#define GST_ENCODER_H264  "nvv4l2h264enc"
#define GST_ENCODER_H265  "nvv4l2h265enc"
#define GST_ENCODER_VP8   "nvv4l2vp8enc"
#define GST_ENCODER_VP9   "nvv4l2vp9enc"
#define GST_ENCODER_MJPEG "nvjpegenc"

#else
	
/**
 * Use OMX codecs for JetPack 4 and older
 * @internal
 * @ingroup codec
 */
#define GST_CODECS_OMX

#if GST_CHECK_VERSION(1,0,0)

// Decoders for JetPack <= 4 and GStreamer >= 1.0
#define GST_DECODER_H264  "omxh264dec"
#define GST_DECODER_H265  "omxh265dec"
#define GST_DECODER_VP8   "omxvp8dec"
#define GST_DECODER_VP9   "omxvp9dec"
#define GST_DECODER_MPEG2 "omxmpeg2videodec"
#define GST_DECODER_MPEG4 "omxmpeg4videodec"
#define GST_DECODER_MJPEG "nvjpegdec"

// Encoders for JetPack <= 4 and GStreamer >= 1.0
#define GST_ENCODER_H264  "omxh264enc"
#define GST_ENCODER_H265  "omxh265enc"
#define GST_ENCODER_VP8   "omxvp8enc"
#define GST_ENCODER_VP9   "omxvp9enc"
#define GST_ENCODER_MJPEG "nvjpegenc"

#else
	
// Decoders for JetPack <= 4 and GStreamer < 1.0
#define GST_DECODER_H264  "nv_omx_h264dec"
#define GST_DECODER_H265  "nv_omx_h265dec"
#define GST_DECODER_VP8   "nv_omx_vp8dec"
#define GST_DECODER_VP9   "nv_omx_vp9dec"
#define GST_DECODER_MPEG2 "nx_omx_mpeg2videodec"
#define GST_DECODER_MPEG4 "nx_omx_mpeg4videodec"
#define GST_DECODER_MJPEG "nvjpegdec"

// Encoders for JetPack <= 4 and GStreamer < 1.0
#define GST_ENCODER_H264  "nv_omx_h264enc"
#define GST_ENCODER_H265  "nv_omx_h265enc"
#define GST_ENCODER_VP8   "nv_omx_vp8enc"
#define GST_ENCODER_VP9   "nv_omx_vp9enc"
#define GST_ENCODER_MJPEG "nvjpegenc"

#endif
#endif

#elif defined(__x86_64)

#if GST_CHECK_VERSION(1,0,0)

// Decoders for x86 and GStreamer >= 1.0
#define GST_DECODER_H264  "avdec_h264"
#define GST_DECODER_H265  "avdec_h265"
#define GST_DECODER_VP8   "vp8dec"
#define GST_DECODER_VP9   "vp9dec"
#define GST_DECODER_MPEG2 "avdec_mpeg2video"
#define GST_DECODER_MPEG4 "avdec_mpeg4"
#define GST_DECODER_MJPEG "jpegdec"

// Encoders for x86 and GStreamer >= 1.0
#define GST_ENCODER_H264  "x264enc"
#define GST_ENCODER_H265  "x265enc"
#define GST_ENCODER_VP8   "vp8enc"
#define GST_ENCODER_VP9   "vp9enc"
#define GST_ENCODER_MJPEG "jpegenc"

#endif
#endif
#endif
