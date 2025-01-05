load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "image",
    srcs = [
        "imageIO.cpp",
        "imageLoader.cpp",
        "imageWriter.cpp",
    ],
    copts = [
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":image_format",
        ":image_includes",
        "//cuda:cuda_lib_headers",
        "//threads",
        "//video:video_includes",
    ],
)

cc_library(
    name = "image_includes",
    srcs = [],
    hdrs = [
        "imageIO.h",
        "imageLoader.h",
        "imageWriter.h",
        "loadImage.h",
        "stb/stb_image.h",
        "stb/stb_image_resize.h",
        "stb/stb_image_write.h",
        "stb/stb_truetype.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "//video:video_options",
    ],
)

cc_library(
    name = "image_format",
    srcs = [],
    hdrs = [
        "imageFormat.h",
        "imageFormat.inl",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "//cuda:cuda_utility",
    ],
)
