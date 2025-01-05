load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "base",
    srcs = glob(["*.cpp"]),
    copts = [
        "-fPIC",
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base_includes",
        "//threads",
    ],
)

cc_library(
    name = "base_includes",
    srcs = [],
    hdrs = glob(["*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
    ],
)
