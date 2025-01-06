cc_library(
    name = "libsoup",
    hdrs = glob([
        "include/libsoup-2.4/**/*.h*",
        "include/nlohmann/**/*.h*",
    ]),
    copts = [
        # "-I/usr/include/libsoup-3.0",
    ],
    includes = [
        "include/libsoup-2.4",
        "include/nlohmann",
    ],
    linkopts = [
        "-l:/usr/lib/x86_64-linux-gnu/libsoup-2.4.so",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@glib",
        "@json_glib",
    ],
)
