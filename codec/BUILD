load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "codec",
    srcs = [
        "gstEncoder.cpp",
        "gstWebRTC.cpp",
    ],
    copts = [
        "-Wno-deprecated-declarations",
        "-Wno-sign-compare",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":codec_includes",
        ":gst_buffer_manager",
        ":gst_decoder",
        ":gst_utility",
        "//cuda:colorspace_lib",
        "//network",
        "//threads",
        "//video:video_source",
        "@gstreamer",
    ],
)

cc_library(
    name = "codec_includes",
    srcs = [],
    hdrs = [
        "gstDecoder.h",
        "gstEncoder.h",
        "gstWebRTC.h",
        "nvbuf_utils.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":gst_utility",
        "//base:base_includes",
    ],
)

cc_library(
    name = "gst_buffer_manager",
    srcs = [
        "gstBufferManager.cpp",
    ],
    hdrs = [
        "gstBufferManager.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":gst_utility",
        "//base",
        "//cuda:colorspace_lib",
    ],
)

cc_library(
    name = "gst_decoder",
    srcs = [
        "gstDecoder.cpp",
    ],
    hdrs = [
        "gstDecoder.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":codec_includes",
        ":gst_buffer_manager",
        ":gst_utility",
        "//base",
        "//network",
    ],
)

cc_library(
    name = "gst_utility",
    srcs = [
        "gstUtility.cpp",
    ],
    hdrs = [
        "gstUtility.h",
        "nvbuf_utils.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "//video:video_options",
        "@gstreamer",
    ],
)
