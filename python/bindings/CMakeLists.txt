# Set CMake policies to suppress deprecation warnings
cmake_policy(SET CMP0148 NEW)

# clear CMakeCache of Python version
unset(PYTHONINTERP_FOUND CACHE)
unset(PYTHON_EXECUTABLE CACHE)
unset(PYTHON_VERSION_STRING CACHE)
unset(PYTHON_VERSION_MAJOR CACHE)
unset(PYTHON_VERSION_MINOR CACHE)
unset(PYTHON_VERSION_PATCH CACHE)

unset(PYTHON_INCLUDE_PATH CACHE)
unset(PYTHON_INCLUDE_DIRS CACHE)
unset(PYTHON_INCLUDE_DIR CACHE)
unset(PYTHON_LIBRARY CACHE)
unset(PYTHON_LIBRARIES CACHE)
unset(PYTHON_DEBUG_LIBRARIES CACHE)
unset(PYTHON_MODULE_PREFIX CACHE)
unset(PYTHON_MODULE_EXTENSION CACHE)

unset(PYTHONLIBS_FOUND CACHE)
unset(PYTHONLIBS_VERSION_STRING CACHE)

# locate requested python version
message("-- detecting Python ${PYTHON_BINDING_VERSION}...")

# Try to find Python using modern CMake detection
find_package(Python${PYTHON_BINDING_VERSION} COMPONENTS Interpreter Development QUIET)

# If not found, try alternative detection methods for uv environments
if(NOT Python${PYTHON_BINDING_VERSION}_FOUND)
	# Try to find Python executable directly
	find_program(PYTHON_EXECUTABLE python${PYTHON_BINDING_VERSION} PATHS /opt/env/bin /usr/bin /usr/local/bin)
	
	if(PYTHON_EXECUTABLE)
		# Get Python version and paths
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" OUTPUT_VARIABLE PYTHON_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)" OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.exec_prefix)" OUTPUT_VARIABLE PYTHON_EXEC_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
		
		# Set version components
		string(REPLACE "." ";" PYTHON_VERSION_LIST ${PYTHON_VERSION_STRING})
		list(GET PYTHON_VERSION_LIST 0 PYTHON_VERSION_MAJOR)
		list(GET PYTHON_VERSION_LIST 1 PYTHON_VERSION_MINOR)
		
		# Set include and library paths
		set(PYTHON_INCLUDE_DIRS "${PYTHON_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
		set(PYTHON_LIBRARIES "${PYTHON_EXEC_PREFIX}/lib/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.so")
		
		# Check if paths exist
		if(EXISTS ${PYTHON_INCLUDE_DIRS} AND EXISTS ${PYTHON_LIBRARIES})
			set(Python${PYTHON_BINDING_VERSION}_FOUND TRUE)
			set(PYTHONLIBS_FOUND TRUE)
			message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (via direct detection)")
			message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
			message("-- found Python library:  ${PYTHON_LIBRARIES}")
		else()
			message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
			return()
		endif()
	else()
		message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
		return()
	endif()
endif()

# Set variables for compatibility with existing code
if(Python${PYTHON_BINDING_VERSION}_FOUND)
	# Only set variables if they haven't been set by direct detection
	if(NOT PYTHON_EXECUTABLE)
		set(PYTHON_EXECUTABLE ${Python${PYTHON_BINDING_VERSION}_EXECUTABLE})
		set(PYTHON_INCLUDE_DIRS ${Python${PYTHON_BINDING_VERSION}_INCLUDE_DIRS})
		set(PYTHON_LIBRARIES ${Python${PYTHON_BINDING_VERSION}_LIBRARIES})
		set(PYTHON_VERSION_MAJOR ${Python${PYTHON_BINDING_VERSION}_VERSION_MAJOR})
		set(PYTHON_VERSION_MINOR ${Python${PYTHON_BINDING_VERSION}_VERSION_MINOR})
		set(PYTHONLIBS_FOUND TRUE)

		message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
		message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
		message("-- found Python library:  ${PYTHON_LIBRARIES}")
	endif()
endif() 

include_directories(${PYTHON_INCLUDE_DIRS})

# attempt to find numpy
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
message("-- CMake module path:  ${CMAKE_MODULE_PATH}")

# Try to find NumPy using our detected Python executable
if(PYTHON_EXECUTABLE)
	message("-- checking for NumPy with Python: ${PYTHON_EXECUTABLE}")
	
	# First, let's check what Python packages are available
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('Python path:'); [print(p) for p in sys.path]" 
		OUTPUT_VARIABLE PYTHON_PATH_OUTPUT 
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE PYTHON_PATH_RESULT)
	
	if(PYTHON_PATH_RESULT EQUAL 0)
		message("-- Python path: ${PYTHON_PATH_OUTPUT}")
	endif()
	
	# Try to find NumPy
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy as n; print(n.__version__); print(n.get_include())" 
		OUTPUT_VARIABLE NUMPY_OUTPUT 
		ERROR_VARIABLE NUMPY_ERROR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE NUMPY_RESULT)
	
	if(NUMPY_RESULT EQUAL 0)
		# Parse NumPy version and include directory
		string(REPLACE "\n" ";" NUMPY_LINES ${NUMPY_OUTPUT})
		list(GET NUMPY_LINES 0 NUMPY_VERSION)
		list(GET NUMPY_LINES 1 NUMPY_INCLUDE_DIR)
		
		set(NUMPY_FOUND TRUE)
		message("-- found NumPy version:  ${NUMPY_VERSION}")
		message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
		
		add_definitions(-DHAS_NUMPY)
		include_directories(${NUMPY_INCLUDE_DIR})
		link_directories(${NUMPY_INCLUDE_DIR}/../lib)
	else()
		message("-- NumPy not found")
		message("-- NumPy error: ${NUMPY_ERROR}")
		
		# Try to find NumPy in common locations
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages())" 
			OUTPUT_VARIABLE SITE_PACKAGES 
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE SITE_RESULT)
		
		if(SITE_RESULT EQUAL 0)
			message("-- site-packages: ${SITE_PACKAGES}")
		endif()
	endif()
else()
	# Fallback to standard CMake NumPy detection
	find_package(NumPy QUIET)
	
	if(${NUMPY_FOUND})
		message("-- found NumPy version:  ${NUMPY_VERSION}")
		message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")

		add_definitions(-DHAS_NUMPY)
		include_directories(${NUMPY_INCLUDE_DIR})
		link_directories(${NUMPY_INCLUDE_DIR}/../lib)
	else()
		message("-- NumPy not found")
	endif()
endif()

# build the bindings
file(GLOB pythonUtilsSources *.cpp)

cuda_add_library(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} SHARED ${pythonUtilsSources})
target_link_libraries(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} jetson-utils ${PYTHON_LIBRARIES})

if(${NUMPY_FOUND})
	if(DEFINED NUMPY_VERSION AND NUMPY_VERSION VERSION_GREATER_EQUAL "2.0")
		message("-- NumPy version ${NUMPY_VERSION} detected (>= 2.0) - skipping npymath linkage")
    else()
		target_link_libraries(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} npymath)
    endif()
endif()



set_target_properties(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
				  PROPERTIES
					PREFIX ""
					OUTPUT_NAME "jetson_utils_python"
					LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# on x86, install under /opt/conda/lib/pythonX.X/site-packages
# otherwise, install under /usr/lib/pythonX.X/dist-packages
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
	set(PYTHON_BINDING_INSTALL_DIR /opt/conda/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
else()
	set(PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages)
endif()

install(TARGETS jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../jetson_utils DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
