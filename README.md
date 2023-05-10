# jetson-utils
C++/CUDA/Python multimedia utilities for NVIDIA Jetson:

|                        |                                                 |
|------------------------|-------------------------------------------------|
| [`/`](/)               | Filesystem, CSV/JSON/XML parsing, command-line  |
| [`camera/`](camera/)   | GStreamer-based camera capture (V4L2, MIPI CSI) |
| [`codec/`](codec/)     | GStreamer-based hardware video encoder/decoder  |
| [`cuda/`](cuda/)       | CUDA image processing functions                 |
| [`display/`](display/) | OpenGL window & rendering                       |
| [`image/`](image/)     | Image loading & saving                          |
| [`input/`](input/)     | Human Interface Devices (HID) from `/dev/input` |
| [`network/`](network/) | Sockets, IPv4/IPv6, WebRTC/RTSP server          |
| [`python/`](python/)   | Python bindings and examples                    |
| [`threads/`](threads/) | Multithreading, locks, and events               |
| [`video/`](video/)     | Video streaming interfaces                      |


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