cc_binary(
    name = "v4l2_display",
    srcs = [
        "v4l2-display.cpp",
    ],
    linkopts = [
        "-lGL",
        "-lGLEW",
        "-lGLU",
        "-lX11",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base",
        "//camera",
        "//display",
    ],
)
