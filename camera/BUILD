load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "camera",
    srcs = [
        "gstCamera.cpp",
        "v4l2Camera.cpp",
    ],
    copts = [
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_includes",
        "//cuda:colorspace_lib",
        #"//video:video_includes",
        "//video:video_source",
    ],
)

cc_library(
    name = "camera_includes",
    srcs = [],
    hdrs = [
        "gstCamera.h",
        "v4l2Camera.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "//codec:codec_includes",
        "//codec:gst_buffer_manager",
        "//threads",
    ],
)
