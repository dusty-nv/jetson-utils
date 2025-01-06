_workspace_name = "test"

workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

new_local_repository(
    name = "libsoup",
    build_file = "@//buildfiles:third_party/libsoup.BUILD",
    path = "/usr",
)

git_repository(
    name = "rules_cuda",
    # v0.2.3 breaks some lubcupti for our version of bazel
    commit = "3f2429254ec956220557e79ea9d5f5e8871c2907",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()

new_git_repository(
    name = "glib",
    build_file = "//buildfiles:third_party/glib.BUILD",
    commit = "763cc3b238398614c20069fd67642730e3a6519b",
    patch_cmds = [
        "meson setup builddir -Ddebug=true -Dprefix=$(pwd)/external",
        "ninja -C builddir install",
    ],
    remote = "https://github.com/GNOME/glib.git",
)

new_local_repository(
    name = "json_glib",
    build_file = "@//buildfiles:third_party/json_glib.BUILD",
    path = "/usr",
)

new_git_repository(
    name = "gstreamer",
    build_file = "//buildfiles:third_party/gstreamer.BUILD",
    commit = "2d8273151571fcab887cc81de48e87aeb61b5c06",
    patch_cmds = [
        "meson setup builddir -Ddebug=true -Dprefix=$(pwd)/external",
        "ninja -C builddir install",
    ],
    remote = "https://github.com/GStreamer/gstreamer.git",
    shallow_since = "1712695735 +0100",
)
