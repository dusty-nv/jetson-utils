
cmake_minimum_required(VERSION 3.5)

# determine if jetson-utils is being built as a submodule inside another repo,
# or if it's being build standalone (if the later, we need to do some configuration)
get_directory_property(hasParent PARENT_DIRECTORY)

if(hasParent)
	message("-- jetson-utils:  building as submodule, ${hasParent}")

	# this was originally meant for building with nvidia inference stuff and not needing to setup the cuda directories. We aren't doing that, so set them up.
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
	find_package(CUDA)
	message("-- CUDA version: ${CUDA_VERSION}")

	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

	if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

		if(CUDA_VERSION_MAJOR GREATER 9)
			message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
			set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
		endif()

		if(CUDA_VERSION_MAJOR GREATER 10)
			message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
			set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
		endif()
	endif()

	# Use CMAKE_CURRENT_BINARY_DIR so it doesn't pollute the root project's binary dir
	set(JETSON_UTILS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}")
	set(JETSON_UTILS_INCLUDE_DIR "${JETSON_UTILS_OUTPUT_DIR}/include")

	# Update the global-style variables for internal jetson-utils use
	set(PROJECT_OUTPUT_DIR  ${JETSON_UTILS_OUTPUT_DIR})
	set(PROJECT_INCLUDE_DIR ${JETSON_UTILS_INCLUDE_DIR})

	# Ensure the directories exist
	file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

else()
	message("-- jetson-utils:  building as standalone")
	
	# standalone project
	project(jetson-utils)
  
	# -std=gnu++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")	
  
	# setup CUDA
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
	find_package(CUDA)
	message("-- CUDA version: ${CUDA_VERSION}")

	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

	if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

		if(CUDA_VERSION_MAJOR GREATER 9)
			message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
			set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
		endif()

		if(CUDA_VERSION_MAJOR GREATER 10)
			message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
			set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
		endif()
	endif()
	
	# setup project output paths
	set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
	set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

	file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

	message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
	message("-- output path:  ${PROJECT_OUTPUT_DIR}")

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
	
	# detect distro version
	find_program(LSB_RELEASE_EXEC lsb_release)

	execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename OUTPUT_VARIABLE LSB_RELEASE_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)

	message("-- distro ID:       ${LSB_RELEASE_ID}")
	message("-- distro version:  ${LSB_RELEASE_VERSION}")
	message("-- distro codename: ${LSB_RELEASE_CODENAME}")

	# enable includes like <jetson_utils/xyz.h>
	include_directories(${PROJECT_INCLUDE_DIR})

endif()

find_package(CUDA REQUIRED)

include_directories(
	${CUDA_INCLUDE_DIRS}
	/usr/local/include/jetson-utils
	/usr/local/include/jetson-inference
)

# --- TensorRT Discovery ---
# Allow users to pass -DTENSORRT_ROOT=/path/to/TRT via command line
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT SDK root")

# Define search hints for different environments
set(TRT_HINTS
    ${TENSORRT_ROOT}
    ${CMAKE_SOURCE_DIR}/../tensorrt  # Common if vendored
    /opt/tensorrt-10.3
	/opt/tensorrt-10.13				# support both versions of tensorrt
    /usr/lib/x86_64-linux-gnu       # Standard x86
    /usr/lib/aarch64-linux-gnu      # Standard Jetson
)

# 1. Search for headers
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_HINTS}
    PATH_SUFFIXES include
)

# 2. Search for the library (Handles the nv_infer vs nvinfer name mismatch)
find_library(TENSORRT_LIB 
    NAMES nvinfer nv_infer
    HINTS ${TRT_HINTS}
    PATH_SUFFIXES lib lib64
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
    message(STATUS "-- Found TensorRT: ${TENSORRT_LIB}")
    include_directories(${TENSORRT_INCLUDE_DIR})
    set(TRT_LIBRARIES ${TENSORRT_LIB})
else()
    # Fallback/Failure
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT or install dependencies.")
endif()

# option for enabling/disabling NVMM memory in multimedia stack
find_library(NVBUF_UTILS NAMES nvbuf_utils PATHS /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/tegra)
message("-- nvbuf_utils:  ${NVBUF_UTILS}")

if(NVBUF_UTILS)
	set(ENABLE_NVMM_DEFAULT ON)
else()
	set(ENABLE_NVMM_DEFAULT OFF)
endif()

option(ENABLE_NVMM "Enable use of NVMM zero-copy memory in video and camera streaming" ${ENABLE_NVMM_DEFAULT})
message("-- NVMM zero-copy memory:  ENABLE_NVMM=${ENABLE_NVMM}")

if(ENABLE_NVMM)
	add_definitions(-DENABLE_NVMM)
endif()

# additional paths for includes and libraries
include_directories(${PROJECT_INCLUDE_DIR}/jetson-utils)

include_directories(
	/usr/include/gstreamer-1.0 
	/usr/include/libxml2  /usr/include/libsoup-2.4 
	/usr/include/glib-2.0 /usr/include/json-glib-1.0
	/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/gstreamer-1.0/include 
	/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/glib-2.0/include/
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	include_directories(/usr/src/jetson_multimedia_api/include)
	link_directories(/usr/lib/aarch64-linux-gnu/tegra)
endif()

# build library
file(GLOB jetsonUtilsSrc cpp/*/*.cpp cuda/*.cpp cuda/*.cu)
file(GLOB jetsonUtilsInc cpp/*/*.h cpp/*/*.hpp cpp/*/*.inl cuda/*.h cuda/*.cuh)

cuda_add_library(jetson-utils SHARED ${jetsonUtilsSrc})

target_include_directories(jetson-utils PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries( jetson-utils 
	GL GLU GLEW ${CUDA_nppicc_LIBRARY}
	gstreamer-1.0 gstapp-1.0 gstpbutils-1.0 
	gstwebrtc-1.0 gstsdp-1.0 gstrtspserver-1.0 
	json-glib-1.0 soup-2.4 ${TRT_LIBRARIES}
)

# Fix for C++20/C++23 where <cstdint> is no longer implicit
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_definitions(-include cstdint)
endif()

if(NVBUF_UTILS)
	target_link_libraries(jetson-utils nvbuf_utils)
endif()

# transfer all headers to the include directory 
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/jetson-utils)

foreach(include ${jetsonUtilsInc})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR}/jetson-utils COPYONLY)
endforeach()
	
# install headers
foreach(include ${jetsonUtilsInc})
	install(FILES "${include}" DESTINATION include/jetson-utils)
endforeach()

# install the shared library
install(TARGETS jetson-utils DESTINATION lib EXPORT jetson-utilsConfig)

# install the cmake project, for importing
install(EXPORT jetson-utilsConfig DESTINATION share/jetson-utils/cmake)

# build python bindings
# add_subdirectory(python)

# build sample apps
add_subdirectory(cpp/video/video-viewer)

#add_subdirectory(camera/camera-viewer)
#add_subdirectory(display/gl-display-test)
#add_subdirectory(network/webrtc-server)
#add_subdirectory(network/rtsp-server)

