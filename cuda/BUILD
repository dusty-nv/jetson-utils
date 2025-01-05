load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cc_library(
    name = "cuda_lib",
    srcs = [
        "cudaPointCloud.cpp",
    ],
    copts = [
        "-fopenmp",
        "-Wall",
        "-std=c++17",
        "-fPIC",
        "-Wno-error=deprecated-declarations",
        "-DWITH_CUDA",
    ],
    linkopts = [
        "-L/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        "-lnppig_static",
        "-lnppc_static",
        "-lnppif_static",
        "-lnppicc_static",
        "-lnvbufsurface",
        "-lnvbufsurftransform",
        "-lculibos",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":colorspace_lib",
        ":cuda_cuda_lib",
        "//display:display_includes",
        "//display:gl_buffer",
        "//display:gl_utility",
    ],
)

cc_library(
    name = "colorspace_lib",
    srcs = [
        "cudaBayer.cpp",
        "cudaColorspace.cpp",
    ],
    copts = [
        "-fopenmp",
        "-Wall",
        "-std=c++17",
        "-fPIC",
        "-Wno-error=deprecated-declarations",
        "-DWITH_CUDA",
    ],
    linkopts = [
        "-L/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        "-lnppig_static",
        "-lnppc_static",
        "-lnppif_static",
        "-lnppicc_static",
        "-lnvbufsurface",
        "-lnvbufsurftransform",
        "-lculibos",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":colorspace_cuda_lib",
        ":cuda_cuda_lib",
        "//display:display_includes",
        "//display:gl_buffer",
        "//display:gl_utility",
    ],
)

cuda_library(
    name = "cuda_cuda_lib",
    srcs = [
        "cudaCrop.cu",
        "cudaDraw.cu",
        "cudaFilterMode.cu",
        "cudaFont.cu",
        "cudaNormalize.cu",
        "cudaOverlay.cu",
        "cudaPointCloud.cu",
        "cudaResize.cu",
        "cudaWarp-affine.cu",
        "cudaWarp-fisheye.cu",
        "cudaWarp-intrinsic.cu",
    ],
    hdrs = [
        "cudaAlphaBlend.cuh",
    ],
    copts = [
        "-std=c++17",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-g",
        "-g",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":colorspace_cuda_lib",
        ":cuda_lib_headers",
        "//image",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "colorspace_cuda_lib",
    srcs = [
        "cudaColormap.cu",
        "cudaGrayscale.cu",
        "cudaRGB.cu",
        "cudaYUV-NV12.cu",
        "cudaYUV-YUYV.cu",
        "cudaYUV-YV12.cu",
    ],
    hdrs = [
        "cudaFilterMode.cuh",
    ],
    copts = [
        "-std=c++17",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-g",
        "-g",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//image",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "cuda_lib_headers",
    hdrs = [
        "cudaBayer.h",
        "cudaColormap.h",
        "cudaColorspace.h",
        "cudaCrop.h",
        "cudaDraw.h",
        "cudaFilterMode.h",
        "cudaFont.h",
        "cudaGrayscale.h",
        "cudaMappedMemory.h",
        "cudaMath.h",
        "cudaNormalize.h",
        "cudaOverlay.h",
        "cudaPointCloud.h",
        "cudaRGB.h",
        "cudaResize.h",
        "cudaUtility.h",
        "cudaVector.h",
        "cudaWarp.h",
        "cudaYUV.h",
    ],
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_utility",
        "//base:base_includes",
        "//image:image_format",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "cuda_utility",
    hdrs = ["cudaUtility.h"],
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "cuda_mapped_memory",
    hdrs = ["cudaMappedMemory.h"],
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "@local_cuda//:cuda_runtime",
    ],
)
