#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import queue

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

class VideoSink:

    def __init__(self, logger, filename, width, height, framerate):
        self.__logger = logger
        pipeline, appsrc = self.__build_pipeline(filename, width, height, framerate)
        self.__pipeline = pipeline
        self.__appsrc = appsrc

    def stop(self):
        if self.__appsrc:
            self.__appsrc.emit("end-of-stream")

    def push_buffer(self, buffer):
        if self.__appsrc:
            self.__appsrc.emit("push-buffer", buffer)

    def __build_pipeline(self, filename, width, height, framerate):
        pipeline = Gst.Pipeline.new("video-sink")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.__on_message)

        caps = Gst.Caps("video/x-raw,format=RGBA,width=%d,height=%d,framerate=%s"
                        % (width, height, framerate))
        appsrc = Gst.ElementFactory.make("appsrc")
        appsrc.set_property("caps", caps)
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("format", Gst.Format.TIME)
        pipeline.add(appsrc)

        # queue.
        queue = Gst.ElementFactory.make("queue")
        queue.set_property("max-size-buffers", 2)
        queue.set_property("leaky", 2) # leaky downstream
        pipeline.add(queue)

        # copy to NVMM memory.
        videoconvert = Gst.ElementFactory.make("nvvidconv")
        pipeline.add(videoconvert)

        caps = Gst.Caps("video/x-raw(memory:NVMM),format=NV12")
        convcaps = Gst.ElementFactory.make("capsfilter")
        convcaps.set_property("caps", caps)
        pipeline.add(convcaps)

        # make a perfect stream.
        videorate = Gst.ElementFactory.make("videorate")
        pipeline.add(videorate)

        caps = Gst.Caps("video/x-raw(memory:NVMM),framerate=%s" % framerate)
        ratecaps = Gst.ElementFactory.make("capsfilter")
        ratecaps.set_property("caps", caps)
        pipeline.add(ratecaps)

        # H.264 encoder
        encoder = Gst.ElementFactory.make("nvv4l2h264enc")
        pipeline.add(encoder)

        # H.264 parser
        parser = Gst.ElementFactory.make("h264parse")
        pipeline.add(parser)

        # Muxer
        muxer = Gst.ElementFactory.make("qtmux")
        pipeline.add(muxer)

        # File sink
        filesink = Gst.ElementFactory.make("filesink")
        filesink.set_property("location", filename)
        pipeline.add(filesink)

        appsrc.link(queue)
        queue.link(videoconvert)
        videoconvert.link(convcaps)
        convcaps.link(videorate)
        videorate.link(ratecaps)
        ratecaps.link(encoder)
        encoder.link(parser)
        parser.link(muxer)
        muxer.link(filesink)

        pipeline.set_state(Gst.State.PLAYING)

        return pipeline, appsrc

    def __on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            if self.__pipeline:
                self.__pipeline.set_state(Gst.State.NULL)
                self.__pipeline = None
        elif t == Gst.MessageType.ERROR:
            self.stop()
            err, debug = message.parse_error()
            self.__logger.error("Error: %s (%s)" % (err, debug))
