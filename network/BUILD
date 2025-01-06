load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "network",
    srcs = [
        "RTSPServer.cpp",
        "WebRTCServer.cpp",
    ],
    hdrs = [
        "RTSPServer.h",
        "WebRTCServer.h",
    ],
    copts = [
        "-Wno-deprecated-declarations",
        "-Wno-sign-compare",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":network_basic",
        "//base",
        "//codec:gst_utility",
        "//threads",
        "@libsoup",
    ],
)

cc_library(
    name = "network_basic",
    srcs = [
        "IPv4.cpp",
        "IPv6.cpp",
        "Networking.cpp",
        "Socket.cpp",
        "URI.cpp",
    ],
    hdrs = [
        "Endian.h",
        "IPv4.h",
        "IPv6.h",
        "Networking.h",
        "Socket.h",
        "URI.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base",
    ],
)
