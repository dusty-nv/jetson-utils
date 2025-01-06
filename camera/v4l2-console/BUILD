cc_binary(
    name = "v4l2_console",
    srcs = [
        "v4l2-console.cpp",
    ],
    linkopts = [
        "-lGL",
        "-lGLEW",
        "-lGLU",
        "-lX11",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//camera",
    ],
)
