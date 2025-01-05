load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "video_output",
    srcs = [
        "videoOutput.cpp",
    ],
    hdrs = [
        "videoOutput.h",
    ],
    copts = [
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":video_options",
        "//threads",
    ],
)

cc_library(
    name = "video_source",
    srcs = [
        "videoSource.cpp",
    ],
    hdrs = [
        "videoSource.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":video_options",
        "//base:base_includes",
        "//camera:camera_includes",
        "//image",
    ],
)

cc_library(
    name = "video_source_create",
    srcs = [
        "videoSourceCreate.cpp",
    ],
    hdrs = [
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":video_options",
        ":video_source",
        "//base:base_includes",
        "//camera",
        "//codec:gst_decoder",
        "//image",
    ],
)

cc_library(
    name = "video_output_create",
    srcs = [
        "videoOutputCreate.cpp",
    ],
    hdrs = [
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":video_options",
        ":video_output",
        "//base:base_includes",
        "//camera",
        "//codec",
        "//display",
        "//image",
    ],
)

cc_library(
    name = "video_includes",
    srcs = [
    ],
    hdrs = [
        "videoOutput.h",
        "videoSource.h",
    ],
    copts = [
        "-std=c++17",
        "-Wno-sign-compare",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":video_options",
        "//base:base_includes",
        "//network:network_basic",
    ],
)

cc_library(
    name = "video_options",
    srcs = [
        "videoOptions.cpp",
    ],
    hdrs = [
        "videoOptions.h",
    ],
    copts = [
        "-std=c++17",
        "-Wno-sign-compare",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base",
        "//image:image_format",
        "//network:network_basic",
    ],
)
