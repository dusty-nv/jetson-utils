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

#endif

