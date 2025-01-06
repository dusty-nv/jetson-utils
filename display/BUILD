load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "display",
    srcs = glob(["*.cpp"]),
    copts = [
        "-Wno-deprecated-declarations",
        "-Wno-sign-compare",
    ],
    linkopts = [
        "-lGL",
        "-lGLU",
        "-lGLEW",
        "-lX11",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":display_includes",
        ":gl_buffer",
        ":gl_utility",
        "//base",
        "//cuda:cuda_lib",
        "//network",
        "//video:video_output",
    ],
)

cc_library(
    name = "display_includes",
    srcs = [],
    hdrs = glob(["*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
    ],
)

cc_library(
    name = "gl_utility",
    srcs = [
    ],
    hdrs = [
        "glUtility.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "gl_buffer",
    srcs = [
    ],
    hdrs = [
        "glBuffer.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
    ],
)
