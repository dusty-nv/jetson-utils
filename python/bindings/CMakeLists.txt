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
message("-- PYTHON_BINDING_VERSION: ${PYTHON_BINDING_VERSION}")

# First try to find Python in the virtual environment
# Check for the specific version requested
if(EXISTS "/opt/venv/bin/python${PYTHON_BINDING_VERSION}")
	set(PYTHON_EXECUTABLE "/opt/venv/bin/python${PYTHON_BINDING_VERSION}")
	message("-- found Python executable in virtual environment: ${PYTHON_EXECUTABLE}")
else()
	# Fallback to general search
	find_program(PYTHON_EXECUTABLE python${PYTHON_BINDING_VERSION} PATHS /opt/venv/bin /opt/env/bin /usr/bin /usr/local/bin)
	if(PYTHON_EXECUTABLE)
		message("-- found Python executable: ${PYTHON_EXECUTABLE}")
	endif()
endif()

if(PYTHON_EXECUTABLE)
	# Set the Python executable for CMake's detection
	set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
	
	# Debug: Test the Python executable directly
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('Python version:', sys.version); print('Python executable:', sys.executable); print('Python prefix:', sys.prefix)" 
		OUTPUT_VARIABLE PYTHON_DEBUG_OUTPUT 
		ERROR_VARIABLE PYTHON_DEBUG_ERROR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE PYTHON_DEBUG_RESULT)
	
	if(PYTHON_DEBUG_RESULT EQUAL 0)
		message("-- Python debug info: ${PYTHON_DEBUG_OUTPUT}")
	else()
		message("-- Python debug failed: ${PYTHON_DEBUG_ERROR}")
	endif()
	
	# Debug: Test NumPy directly
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print('NumPy version:', numpy.__version__); print('NumPy location:', numpy.__file__)" 
		OUTPUT_VARIABLE NUMPY_DEBUG_OUTPUT 
		ERROR_VARIABLE NUMPY_DEBUG_ERROR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE NUMPY_DEBUG_RESULT)
	
	if(NUMPY_DEBUG_RESULT EQUAL 0)
		message("-- NumPy debug info: ${NUMPY_DEBUG_OUTPUT}")
	else()
		message("-- NumPy debug failed: ${NUMPY_DEBUG_ERROR}")
	endif()
endif()

# Try to find Python using modern CMake detection
message("-- Trying to find Python${PYTHON_BINDING_VERSION} with CMake...")
find_package(Python${PYTHON_BINDING_VERSION} COMPONENTS Interpreter Development QUIET)
message("-- Python${PYTHON_BINDING_VERSION}_FOUND: ${Python${PYTHON_BINDING_VERSION}_FOUND}")

# If not found, try alternative detection methods for uv environments
if(NOT Python${PYTHON_BINDING_VERSION}_FOUND)
	message("-- CMake Python detection failed, trying direct detection...")
	# Try to find Python executable directly
	find_program(PYTHON_EXECUTABLE python${PYTHON_BINDING_VERSION} PATHS /opt/venv/bin /opt/env/bin /usr/bin /usr/local/bin)
	
	if(PYTHON_EXECUTABLE)
		message("-- Found Python executable: ${PYTHON_EXECUTABLE}")
		# Get Python version and paths
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" OUTPUT_VARIABLE PYTHON_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)" OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.exec_prefix)" OUTPUT_VARIABLE PYTHON_EXEC_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
		message("-- Python version string: ${PYTHON_VERSION_STRING}")
		message("-- Python prefix: ${PYTHON_PREFIX}")
		message("-- Python exec prefix: ${PYTHON_EXEC_PREFIX}")
		
		# Set version components
		string(REPLACE "." ";" PYTHON_VERSION_LIST ${PYTHON_VERSION_STRING})
		list(GET PYTHON_VERSION_LIST 0 PYTHON_VERSION_MAJOR)
		list(GET PYTHON_VERSION_LIST 1 PYTHON_VERSION_MINOR)
		message("-- Python version major: ${PYTHON_VERSION_MAJOR}")
		message("-- Python version minor: ${PYTHON_VERSION_MINOR}")
		
		# Try multiple possible include and library paths
		set(PYTHON_INCLUDE_CANDIDATES
			"${PYTHON_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
			"${PYTHON_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m"
			"${PYTHON_EXEC_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
			"${PYTHON_EXEC_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m"
			"/usr/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
			"/usr/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m"
		)
		
		set(PYTHON_LIBRARY_CANDIDATES
			"${PYTHON_EXEC_PREFIX}/lib/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.so"
			"${PYTHON_EXEC_PREFIX}/lib/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m.so"
			"${PYTHON_PREFIX}/lib/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.so"
			"${PYTHON_PREFIX}/lib/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m.so"
			"/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.so"
			"/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m.so"
		)
		
		# Find the first existing include directory
		message("-- Searching for Python include directory...")
		foreach(INCLUDE_CANDIDATE ${PYTHON_INCLUDE_CANDIDATES})
			message("-- Checking include candidate: ${INCLUDE_CANDIDATE}")
			if(EXISTS ${INCLUDE_CANDIDATE})
				set(PYTHON_INCLUDE_DIRS ${INCLUDE_CANDIDATE})
				message("-- Found include directory: ${PYTHON_INCLUDE_DIRS}")
				break()
			endif()
		endforeach()
		
		# Find the first existing library
		message("-- Searching for Python library...")
		foreach(LIBRARY_CANDIDATE ${PYTHON_LIBRARY_CANDIDATES})
			message("-- Checking library candidate: ${LIBRARY_CANDIDATE}")
			if(EXISTS ${LIBRARY_CANDIDATE})
				set(PYTHON_LIBRARIES ${LIBRARY_CANDIDATE})
				message("-- Found library: ${PYTHON_LIBRARIES}")
				break()
			endif()
		endforeach()
		
		# Check if paths exist
		if(EXISTS ${PYTHON_INCLUDE_DIRS} AND EXISTS ${PYTHON_LIBRARIES})
			set(Python${PYTHON_BINDING_VERSION}_FOUND TRUE)
			set(PYTHONLIBS_FOUND TRUE)
			message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (via direct detection)")
			message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
			message("-- found Python library:  ${PYTHON_LIBRARIES}")
		else()
			message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
			message("-- Include dir candidates: ${PYTHON_INCLUDE_CANDIDATES}")
			message("-- Library candidates: ${PYTHON_LIBRARY_CANDIDATES}")
			message("-- Found include: ${PYTHON_INCLUDE_DIRS}")
			message("-- Found library: ${PYTHON_LIBRARIES}")
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

# Use CMake's built-in NumPy detection (available since CMake 3.12)
# This is more reliable than custom detection methods and handles virtual environments properly
if(PYTHON_EXECUTABLE)
	# Set the Python executable for CMake's NumPy detection
	set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
	message("-- using Python executable for NumPy detection: ${PYTHON_EXECUTABLE}")
	
	# Clear any cached Python variables to force detection with our specific Python
	unset(Python_NumPy_FOUND CACHE)
	unset(Python_NumPy_VERSION CACHE)
	unset(Python_NumPy_INCLUDE_DIRS CACHE)
	unset(Python_NumPy_LIBRARIES CACHE)
	
	# Also clear the general Python variables
	unset(Python_FOUND CACHE)
	unset(Python_EXECUTABLE CACHE)
	unset(Python_VERSION CACHE)
	unset(Python_INCLUDE_DIRS CACHE)
	unset(Python_LIBRARIES CACHE)
	
	# Force CMake to use our specific Python executable
	set(Python_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Python executable" FORCE)
	
	# Debug: Check what Python executable CMake is actually using
	message("-- Python_EXECUTABLE after setting: ${Python_EXECUTABLE}")
	message("-- PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
	
	# Use the specific Python version we detected
	find_package(Python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} COMPONENTS NumPy QUIET)
	
	# Debug: Check what Python executable was actually used
	message("-- Python_EXECUTABLE after find_package: ${Python_EXECUTABLE}")
	
	# If CMake's detection failed, try direct NumPy detection with our Python
	if(NOT Python_NumPy_FOUND)
		message("-- CMake NumPy detection failed, trying direct detection with ${PYTHON_EXECUTABLE}")
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.__version__); print(numpy.get_include())" 
			OUTPUT_VARIABLE NUMPY_DIRECT_OUTPUT 
			ERROR_VARIABLE NUMPY_DIRECT_ERROR
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE NUMPY_DIRECT_RESULT)
		
		if(NUMPY_DIRECT_RESULT EQUAL 0)
			# Parse NumPy version and include directory
			string(REPLACE "\n" ";" NUMPY_LINES ${NUMPY_DIRECT_OUTPUT})
			list(GET NUMPY_LINES 0 NUMPY_VERSION)
			list(GET NUMPY_LINES 1 NUMPY_INCLUDE_DIR)
			
			set(NUMPY_FOUND TRUE)
			set(Python_NumPy_FOUND TRUE)
			message("-- found NumPy version:  ${NUMPY_VERSION} (via direct detection)")
			message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
			
			add_definitions(-DHAS_NUMPY)
			include_directories(${NUMPY_INCLUDE_DIR})
			link_directories(${NUMPY_INCLUDE_DIR}/../lib)
		else()
			message("-- direct NumPy detection also failed: ${NUMPY_DIRECT_ERROR}")
		endif()
	endif()
	
	# Debug: Test if the Python executable we're using actually has NumPy
	execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy; print('NumPy version:', numpy.__version__); print('NumPy include:', numpy.get_include())" 
		OUTPUT_VARIABLE NUMPY_TEST_OUTPUT 
		ERROR_VARIABLE NUMPY_TEST_ERROR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE NUMPY_TEST_RESULT)
	
	# Also test with the original PYTHON_EXECUTABLE if it's different
	if(NOT Python_EXECUTABLE STREQUAL PYTHON_EXECUTABLE)
		message("-- also testing with original PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
		execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print('NumPy version:', numpy.__version__); print('NumPy include:', numpy.get_include())" 
			OUTPUT_VARIABLE NUMPY_TEST_OUTPUT2 
			ERROR_VARIABLE NUMPY_TEST_ERROR2
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE NUMPY_TEST_RESULT2)
		
		if(NUMPY_TEST_RESULT2 EQUAL 0)
			message("-- NumPy test with ${PYTHON_EXECUTABLE} succeeded: ${NUMPY_TEST_OUTPUT2}")
			# Use the working Python executable
			set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
		else()
			message("-- NumPy test with ${PYTHON_EXECUTABLE} also failed: ${NUMPY_TEST_ERROR2}")
		endif()
	endif()
	
	if(NUMPY_TEST_RESULT EQUAL 0)
		message("-- NumPy test with ${Python_EXECUTABLE} succeeded: ${NUMPY_TEST_OUTPUT}")
	else()
		message("-- NumPy test with ${Python_EXECUTABLE} failed: ${NUMPY_TEST_ERROR}")
	endif()
else()
	# Fallback to general Python detection
	find_package(Python COMPONENTS NumPy QUIET)
endif()

if(Python_NumPy_FOUND OR Python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}_NumPy_FOUND)
	set(NUMPY_FOUND TRUE)
	if(Python_NumPy_FOUND)
		set(NUMPY_VERSION ${Python_NumPy_VERSION})
		set(NUMPY_INCLUDE_DIR ${Python_NumPy_INCLUDE_DIRS})
	else()
		set(NUMPY_VERSION ${Python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}_NumPy_VERSION})
		set(NUMPY_INCLUDE_DIR ${Python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}_NumPy_INCLUDE_DIRS})
	endif()
	
	message("-- found NumPy version:  ${NUMPY_VERSION}")
	message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
	
	add_definitions(-DHAS_NUMPY)
	include_directories(${NUMPY_INCLUDE_DIR})
	link_directories(${NUMPY_INCLUDE_DIR}/../lib)
else()
	message("-- NumPy not found")
	
	# Fallback to custom detection if CMake's method fails
	if(PYTHON_EXECUTABLE)
		message("-- trying custom NumPy detection...")
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
			message("-- found NumPy version:  ${NUMPY_VERSION} (via custom detection)")
			message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
			
			add_definitions(-DHAS_NUMPY)
			include_directories(${NUMPY_INCLUDE_DIR})
			link_directories(${NUMPY_INCLUDE_DIR}/../lib)
		else()
			message("-- custom NumPy detection also failed: ${NUMPY_ERROR}")
		endif()
	endif()
endif()

# build the bindings
file(GLOB pythonUtilsSources *.cpp)

message("-- Building Python bindings for version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
message("-- Python sources: ${pythonUtilsSources}")
message("-- Python libraries: ${PYTHON_LIBRARIES}")
message("-- Python include dirs: ${PYTHON_INCLUDE_DIRS}")

# Only build if we have valid Python detection
if(PYTHON_EXECUTABLE AND PYTHON_INCLUDE_DIRS AND PYTHON_LIBRARIES)
	message("-- Creating Python binding target...")
	cuda_add_library(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} SHARED ${pythonUtilsSources})
	target_link_libraries(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} jetson-utils ${PYTHON_LIBRARIES})
	message("-- Python binding target created successfully")
else()
	message("-- Skipping Python binding creation - missing required components")
	message("-- PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
	message("-- PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
	message("-- PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
endif()

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
