# jetson-utils
C++/CUDA/Python multimedia utilities for NVIDIA Jetson:

|                        |                                                 |
|------------------------|-------------------------------------------------|

| [`camera/`](cpp/camera/)   | GStreamer-based camera capture (V4L2, MIPI CSI) |
| [`codec/`](cpp/codec/)     | GStreamer-based hardware video encoder/decoder  |
| [`cuda/`](cuda/)       | CUDA image processing functions                 |
| [`display/`](cpp/display/) | OpenGL window & rendering                       |
| [`image/`](cpp/image/)     | Image loading & saving                          |
| [`input/`](cpp/input/)     | Human Interface Devices (HID) from `/dev/input` |
| [`network/`](cpp/network/) | Sockets, IPv4/IPv6, WebRTC/RTSP server          |
| [`parsers/`](cpp/parsers)        | Filesystem, CSV/JSON/XML parsing, command-line  |
| [`python/`](python/)   | Python bindings and examples                    |
| [`threads/`](cpp/threads/) | Multithreading, locks, and events               |
| [`video/`](cpp/video/)     | Video streaming interfaces                      |


### Documentation

Documentation for jetson-utils can be found here:

* [API Reference](https://github.com/dusty-nv/jetson-inference#api-reference)
* [Camera Streaming and Multimedia](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md)
* [Image Manipulation with CUDA](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md)

### Building from Source

jetson-utils is typically built as a submodule of [jetson-inference](https://github.com/dusty-nv/jetson-inference), but it can also be compiled/installed standalone:

``` bash
git clone https://github.com/dusty-nv/jetson-utils
mkdir build
cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```

If you're missing dependencies, run the [`jetson-inference/CMakePreBuild.sh`](https://github.com/dusty-nv/jetson-inference/blob/master/CMakePreBuild.sh) script.