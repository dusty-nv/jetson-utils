# Bazel is only available for amd64 and arm64.

config_setting(
    name = "aarch64-linux-gnu",
    define_values = {"multiarch": "aarch64-linux-gnu"},
)

config_setting(
    name = "x86_64-linux-gnu",
    define_values = {"multiarch": "x86_64-linux-gnu"},
)

cc_library(
    name = "json_glib",
    hdrs = glob([
        "include/json-glib-1.0/**/*.h",
    ]),
    includes = [
        "include/json-glib-1.0",
    ],
    linkopts =
        select({
            ":aarch64-linux-gnu": ["-Llib/aarch64-linux-gnu"],
            ":x86_64-linux-gnu": ["-Llib/x86_64-linux-gnu"],
            "//conditions:default": [],
        }) + [
            "-l:libjson-glib-1.0.so",
        ],
    visibility = ["//visibility:public"],
    deps = [
        "@glib",
    ],
)
