# jetson-utils
C++/CUDA/Python multimedia utilities for NVIDIA Jetson:

|                        |                                                 |
|------------------------|-------------------------------------------------|
| [`cpp/`](cpp/)   | Various system & media utilities with C++ interfaces |
| &nbsp;&nbsp;&nbsp; [`camera/`](cpp/camera/)   | GStreamer-based camera capture (V4L2, MIPI CSI) |
| &nbsp;&nbsp;&nbsp; [`codec/`](cpp/codec/)     | GStreamer-based hardware video encoder/decoder  |
| &nbsp;&nbsp;&nbsp; [`display/`](cpp/display/) | OpenGL window & rendering                       |
| &nbsp;&nbsp;&nbsp; [`image/`](cpp/image/)     | Image loading & saving                          |
| &nbsp;&nbsp;&nbsp; [`input/`](cpp/input/)     | Human Interface Devices (HID) from `/dev/input` |
| &nbsp;&nbsp;&nbsp; [`network/`](cpp/network/) | Sockets, IPv4/IPv6, WebRTC/RTSP server          |
| &nbsp;&nbsp;&nbsp; [`parsers/`](cpp/parsers)        | Filesystem, CSV/JSON/XML parsing, command-line  |
| &nbsp;&nbsp;&nbsp; [`threads/`](cpp/threads/) | Multithreading, locks, and events               |
| &nbsp;&nbsp;&nbsp; [`video/`](cpp/video/)     | Video streaming interfaces                      |
| [`cuda/`](cuda/)       | CUDA image processing functions                 |
| [`docs/`](docs/)       | Collection of Linux commands and links          |
| [`python/`](python/)   | Python utilities, examples, and C++ bindings    |
| [`scripts/`](scripts/) | Standalone shell scripts in Bash or Python      |


### Documentation

Documentation for jetson-utils can be found here:

* [API Reference](https://github.com/dusty-nv/jetson-inference#api-reference)
* [Camera Streaming and Multimedia](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md)
* [Image Manipulation with CUDA](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md)

Assorted links and tips-and-tricks for Linux are kept under [`docs/`](docs/)

### Building from Source (C++/CUDA)

This will build and install the C++/CUDA library (`libjetson-utils.so`) along with the Python extension module:

``` bash
git clone https://github.com/dusty-nv/jetson-utils
cd jetson-utils
mkdir build
cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```

If you're missing dependencies, run the [`jetson-inference/CMakePreBuild.sh`](https://github.com/dusty-nv/jetson-inference/blob/master/CMakePreBuild.sh) script.

### Install with Pip (Python only)

This will install the Python-native modules from [`python/jetson_utils`](/python/jetson_utils) which do not depend on the C++ extension bindings and get installed on top:

```
pip3 install -e /path/to/your/jetson-utils
```

Or directly from GitHub:

```
pip3 install git+https://github.com/dusty-nv/jetson-utils
```