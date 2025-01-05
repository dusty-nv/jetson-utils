load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cc_binary(
    name = "gl_display_test",
    srcs = [
        "gl-display-test.cpp",
    ],
    linkopts = [
        "-lGL",
        "-lGLEW",
        "-lGLU",
        "-lX11",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_lib",
        "//base",
        "//codec",
        "//cuda:cuda_lib",
        "//display",
        "//network",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_lib",
    srcs = glob(["*.cu"]),
    hdrs = glob(["*.h"]),
    copts = [
        "-std=c++17",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-g",
        "-g",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":headers",
        "//cuda:cuda_cuda_lib",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "headers",
    hdrs = [
        "cudaInteropKernels.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_cuda//:cuda_runtime",
    ],
)
