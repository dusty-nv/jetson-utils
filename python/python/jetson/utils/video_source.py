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

class VideoSource:

    def __init__(self, logger, width=1280, height=720, framerate="30/1", format="NV12", sync=True):
        self.__logger = logger
        self.__width = width
        self.__height = height
        self.__framerate = framerate
        self.__format = format
        self.__sync = sync

        self.__pipeline = None
        self.__queue = queue.Queue()

    def load(self, filename):
        if self.__pipeline:
            raise Exception("Already processing a file")
        self.__pipeline = Gst.Pipeline.new("video-source")

        bus = self.__pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.__on_message)

        self.__build_pipeline(filename)
        self.__pipeline.set_state(Gst.State.PLAYING)


    def stop(self):
        if self.__pipeline:
            self.__pipeline.set_state(Gst.State.NULL)
            self.__pipeline = None

    def is_loading(self):
        return self.__pipeline != None

    def sample_next(self):
        return self.__queue.get()

    def sample_done(self):
        self.__queue.task_done()

    def __build_pipeline(self, filename):
        source = Gst.ElementFactory.make("filesrc")
        source.set_property("location", filename)
        self.__pipeline.add(source)

        decodebin = Gst.ElementFactory.make("decodebin")
        decodebin.connect("pad-added", self.__on_pad_added)
        self.__pipeline.add(decodebin)

        source.link(decodebin)

    def __on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.stop()
        elif t == Gst.MessageType.ERROR:
            self.stop()
            err, debug = message.parse_error()
            self.__logger.error("Error: %s (%s)" % (err, debug))

    def __on_pad_added(self, decodebin, pad):
        caps_str = pad.get_current_caps().to_string()
        if caps_str.startswith("video"):
            self.__logger.info("Detected video stream %s, processing..." % caps_str)
        elif caps_str.startswith("audio"):
            self.__logger.info("Detected audio stream %s, ignoring." % caps_str)
            return

        # queue.
        queue = Gst.ElementFactory.make("queue")
        queue.set_property("max-size-buffers", 2)
        queue.set_property("leaky", 2) # leaky downstream
        self.__pipeline.add(queue)

        # scale image and convert in GPU if needed.
        videoscale = Gst.ElementFactory.make("nvvidconv")
        self.__pipeline.add(videoscale)

        caps = "video/x-raw(memory:NVMM),format=%s" % self.__format
        if self.__width > 0:
            caps += ",width=%d" % self.__width
        if self.__height > 0:
            caps += ",height=%d" % self.__height

        scalecaps = Gst.ElementFactory.make("capsfilter")
        scalecaps.set_property("caps", Gst.Caps(caps))
        self.__pipeline.add(scalecaps)

        # copy to main memory.
        videoconvert = Gst.ElementFactory.make("nvvidconv")
        self.__pipeline.add(videoconvert)

        convcaps = Gst.ElementFactory.make("capsfilter")
        convcaps.set_property("caps", Gst.Caps("video/x-raw"))
        self.__pipeline.add(convcaps)

        # now rate so we don't process as many frames.
        videorate = Gst.ElementFactory.make("videorate")
        self.__pipeline.add(videorate)

        caps = "video/x-raw,framerate=%s" % self.__framerate
        ratecaps = Gst.ElementFactory.make("capsfilter")
        ratecaps.set_property("caps", Gst.Caps(caps))
        self.__pipeline.add(ratecaps)

        # finally, add the sample sink.
        appsink = Gst.ElementFactory.make("appsink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 2)
        appsink.set_property("drop", True)
        appsink.set_property("sync", self.__sync)
        appsink.connect("new-sample", self.__on_new_sample)
        self.__pipeline.add(appsink)

        # link everything together.
        sinkpad = videoscale.get_static_pad("sink")
        pad.link(sinkpad)

        queue.link(videoscale)
        videoscale.link(scalecaps)
        scalecaps.link(videoconvert)
        videoconvert.link(convcaps)
        convcaps.link(videorate)
        videorate.link(ratecaps)
        ratecaps.link(appsink)

    def __on_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        self.__queue.put(sample)
        return Gst.FlowReturn.OK
