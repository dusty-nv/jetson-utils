cc_binary(
    name = "camera_viewer",
    srcs = [
        "camera-viewer.cpp",
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
        "//codec",
        "//display",
    ],
)
