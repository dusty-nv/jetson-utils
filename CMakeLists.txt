# jetson-utils -- CMakeLists.txt
cmake_minimum_required(VERSION 2.8)

# CMP0048: The project() command manages VERSION variables
# @ref https://cmake.org/cmake/help/latest/policy/CMP0048.html
cmake_policy(SET CMP0048 NEW)

# CMP0028: Double colon in target name means ALIAS or IMPORTED target
# @ref https://cmake.org/cmake/help/latest/policy/CMP0028.html
cmake_policy(SET CMP0028 NEW)

# Set name of the project
# @ref https://cmake.org/cmake/help/latest/command/project.html
project(jetson-utils LANGUAGES CXX VERSION 1.0.0)

# Include "GNUInstallDirs" module which defines GNU standard installation directories
# @ref https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html
include(GNUInstallDirs)

# Include "CMakePackageConfigHelpers" module which creates package config files for "find_package" command
# @ref https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html
include(CMakePackageConfigHelpers)

# Set CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Set install directory
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  #set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install)
  set(CMAKE_INSTALL_PREFIX /usr/local)
endif()
message(STATUS "jetson-utils: install directory  ${CMAKE_INSTALL_PREFIX}")

# Detect distro version
find_program(LSB_RELEASE_EXEC lsb_release /usr/bin)

if(NOT DEFINED LSB_RELEASE_EXEC)
  message(FATAL_ERROR "-- jetson-utils: Cannot find 'lsb_release' executable. Please install it via 'apt install lsb_release'.")
endif()

execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename OUTPUT_VARIABLE LSB_RELEASE_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "jetson-utils: distro ID       ${LSB_RELEASE_ID}")
message(STATUS "jetson-utils: distro version  ${LSB_RELEASE_VERSION}")
message(STATUS "jetson-utils: distro codename ${LSB_RELEASE_CODENAME}")

# Use C++11 standard if no standard defined
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
message(STATUS "jetson-utils: using C++ standard ${CMAKE_CXX_STANDARD}")

# Setup CUDA
find_package(CUDA)
message(STATUS "jetson-utils: CUDA version  ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "jetson-utils: CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

    if(CUDA_VERSION_MAJOR GREATER 9)
        message(STATUS "jetson-utils: CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
    endif()

    if(CUDA_VERSION_MAJOR GREATER 10)
        message(STATUS "jetson-utils: CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
    endif()
endif()

# Option for enabling/disabling NVMM memory in multimedia stack
find_package(NVBUF_UTILS REQUIRED)
if(NVBUF_UTILS_FOUND)
  set(ENABLE_NVMM_DEFAULT ON)
else()
  set(ENABLE_NVMM_DEFAULT OFF)
endif()

option(ENABLE_NVMM "Enable use of NVMM zero-copy memory in video and camera streaming" ${ENABLE_NVMM_DEFAULT})
message(STATUS "jetson-utils: Enable NVMM zero-copy memory (ENABLE_NVMM): ${ENABLE_NVMM}")

if(ENABLE_NVMM)
    add_definitions(-DENABLE_NVMM)
endif()

# CMP0072: FindOpenGL prefers GLVND by default when available
# @ref https://cmake.org/cmake/help/latest/policy/CMP0072.html
cmake_policy(SET CMP0072 OLD)  # TODO: test it with NEW

# Find OpenGL and related packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# Find other dependent packages
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)
pkg_check_modules(GSTREAMER_PBUTILS REQUIRED IMPORTED_TARGET gstreamer-pbutils-1.0)
pkg_check_modules(GSTREAMER_WEBRTC REQUIRED IMPORTED_TARGET gstreamer-webrtc-1.0)
pkg_check_modules(GSTREAMER_SDP REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0)
pkg_check_modules(GSTREAMER_RTSP_SERVER REQUIRED IMPORTED_TARGET gstreamer-rtsp-server-1.0)
pkg_check_modules(JSON_GLIB REQUIRED IMPORTED_TARGET json-glib-1.0)
pkg_check_modules(SOUP REQUIRED IMPORTED_TARGET libsoup-2.4)

# Setup source and header files
file(GLOB jetsonUtilitySources *.cpp camera/*.cpp codec/*.cpp cuda/*.cu cuda/*.cpp display/*.cpp image/*.cpp input/*.cpp network/*.cpp threads/*.cpp video/*.cpp)
file(GLOB jetsonUtilityIncludes *.h *.hpp camera/*.h codec/*.h cuda/*.h cuda/*.cuh display/*.h image/*.h image/*.inl input/*.h network/*.h threads/*.h threads/*.inl video/*.h)

# Copy all headers to the include directory
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME})
foreach(include ${jetsonUtilityIncludes})
    message(STATUS "Copying ${include}")
    configure_file(${include} ${PROJECT_BINARY_DIR}/include/jetson-utils COPYONLY)
endforeach()

# Create the library
cuda_add_library(${PROJECT_NAME} SHARED ${jetsonUtilitySources} ${jetsonUtilityIncludes})
target_link_libraries(${PROJECT_NAME}
  GLEW::GLEW
  OpenGL::GL
  OpenGL::GLU
  PkgConfig::GLIB
  PkgConfig::GSTREAMER
  PkgConfig::GSTREAMER_APP
  PkgConfig::GSTREAMER_PBUTILS
  PkgConfig::GSTREAMER_WEBRTC
  PkgConfig::GSTREAMER_SDP
  PkgConfig::GSTREAMER_RTSP_SERVER
  PkgConfig::JSON_GLIB
  PkgConfig::SOUP
  ${CUDA_nppicc_LIBRARY}
)

if(NVBUF_UTILS_FOUND)
    target_link_libraries(${PROJECT_NAME} ${NVBUF_UTILS_LIBRARY})
endif()

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)
set_target_properties(${PROJECT_NAME}
  PROPERTIES
    PUBLIC_HEADER "${jetsonUtilityIncludes}"
)

# Install the library and headers while exporting the targets
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# Install exported targets with a namespace
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE "JetsonUtils::"
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)

# Configure package config files using a template
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)

# Generate package config files
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install package config files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)

# Enable/disable python bindings
set(JU_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "Enable jetson-utils python bindings")
if(${JU_ENABLE_PYTHON_BINDINGS})
  add_subdirectory(python)
endif()

# Enable/disable apps
set(JU_ENABLE_CAMERA_VIEWER OFF CACHE BOOL "Enable jetson-utils camera viewer app")
if(${JU_ENABLE_CAMERA_VIEWER})
  add_subdirectory(camera/camera-viewer)
endif()

set(JU_ENABLE_V4L2_DISPLAY OFF CACHE BOOL "Enable jetson-utils v4l2 display app")
if(${JU_ENABLE_V4L2_DISPLAY})
  add_subdirectory(camera/v4l2-display)
endif()

set(JU_ENABLE_V4L2_CONSOLE OFF CACHE BOOL "Enable jetson-utils v4l2 console app")
if(${JU_ENABLE_V4L2_CONSOLE})
  add_subdirectory(camera/v4l2-console)
endif()

set(JU_ENABLE_VIDEO_VIEWER OFF CACHE BOOL "Enable jetson-utils video viewer app")
if(${JU_ENABLE_VIDEO_VIEWER})
  add_subdirectory(video/video-viewer)
endif()

set(JU_ENABLE_GL_DISPLAY_TEST OFF CACHE BOOL "Enable jetson-utils gl display test app")
if(${JU_ENABLE_GL_DISPLAY_TEST})
  add_subdirectory(display/gl-display-test)
endif()

set(JU_ENABLE_WEBRTC_SERVER OFF CACHE BOOL "Enable jetson-utils webrtc server app")
if(${JU_ENABLE_WEBRTC_SERVER})
  add_subdirectory(network/webrtc-server)
endif()

set(JU_ENABLE_RTSP_SERVER OFF CACHE BOOL "Enable jetson-utils rtsp server app")
if(${JU_ENABLE_RTSP_SERVER})
  add_subdirectory(network/rtsp-server)
endif()
