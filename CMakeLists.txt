
cmake_minimum_required(VERSION 2.8)

# determine if jetson-utils is being built as a submodule inside another repo,
# or if it's being build standalone (if the later, we need to do some configuration)
get_directory_property(hasParent PARENT_DIRECTORY)

if(hasParent)
	message("-- jetson-utils:  building as submodule, ${hasParent}")
	
	# configure Qt
	find_package(Qt4 REQUIRED)
	include(${QT_USE_FILE})
	add_definitions(${QT_DEFINITIONS})
else()
	message("-- jetson-utils:  building as standalone")
	
	# standalone project
	project(jetson-utils)
  
	# -std=gnu++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	
  
	# Qt is used to load images (installed by ubuntu-desktop)
	find_package(Qt4 REQUIRED)
	include(${QT_USE_FILE})
	add_definitions(${QT_DEFINITIONS})
	
	# setup CUDA
	find_package(CUDA)
	message("-- CUDA version: ${CUDA_VERSION}")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-O3 
		-gencode arch=compute_37,code=sm_37
		-gencode arch=compute_53,code=sm_53
		-gencode arch=compute_60,code=sm_60
		-gencode arch=compute_61,code=sm_61
		-gencode arch=compute_62,code=sm_62
	)

	if(CUDA_VERSION_MAJOR GREATER 9)
		message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

		set(
			CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS}; 
			-gencode arch=compute_72,code=sm_72
		)
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
	
	# build C/C++ interface
	include_directories(${PROJECT_INCLUDE_DIR})
	#include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

endif()

include_directories(/usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/ /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/lib/x86_64-linux-gnu/glib-2.0/include/)


# build library
file(GLOB jetsonUtilitySources *.cpp camera/*.cpp codec/*.cpp cuda/*.cu display/*.cpp input/*.cpp network/*.cpp threads/*.cpp)
file(GLOB jetsonUtilityIncludes *.h camera/*.h codec/*.h cuda/*.h display/*.h input/*.h network/*.h threads/*.h)

cuda_add_library(jetson-utils SHARED ${jetsonUtilitySources})
target_link_libraries(jetson-utils Qt4::QtGui GL GLEW gstreamer-1.0 gstapp-1.0)	


# transfer all headers to the include directory
foreach(include ${jetsonUtilityIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()
	
# copy image data
file(GLOB imageData data/*.png data/*.jpg)

foreach(image ${imageData})
	message("-- Copying ${image}")
	file(COPY ${image} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	install(CODE "file(COPY ${image} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )" )
endforeach()
	

# install headers
foreach(include ${jetsonUtilityIncludes})
	install(FILES "${include}" DESTINATION include/jetson-utils)
endforeach()

# install the shared library
install(TARGETS jetson-utils DESTINATION lib EXPORT jetson-utilsConfig)

# install the cmake project, for importing
install(EXPORT jetson-utilsConfig DESTINATION share/jetson-utils/cmake)


# build tests/sample executables
add_subdirectory(camera/gst-camera)
add_subdirectory(camera/v4l2-console)
add_subdirectory(camera/v4l2-display)
add_subdirectory(display/gl-display-test)
