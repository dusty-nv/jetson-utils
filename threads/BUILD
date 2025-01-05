cc_library(
    name = "threads",
    srcs = glob(["*.cpp"]),
    hdrs = glob([
        "*.h",
        "*.inl",
    ]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//base:base_includes",
        "//cuda:cuda_mapped_memory",
    ],
)
