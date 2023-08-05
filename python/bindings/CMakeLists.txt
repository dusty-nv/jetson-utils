# Clear CMakeCache of Python version
unset(PYTHONINTERP_FOUND CACHE)
unset(PYTHON_EXECUTABLE CACHE)
unset(PYTHON_VERSION_STRING CACHE)
unset(PYTHON_VERSION_MAJOR CACHE)
unset(PYTHON_VERSION_MINOR CACHE)
unset(PYTHON_VERSION_PATCH CACHE)
# @ref https://github.com/opencv/opencv/blob/master/cmake/OpenCVDetectPython.cmake
unset(PYTHONLIBS_FOUND)
unset(PYTHON_LIBRARIES)
unset(PYTHON_INCLUDE_PATH)
unset(PYTHON_INCLUDE_DIRS)
unset(PYTHON_DEBUG_LIBRARIES)
unset(PYTHONLIBS_VERSION_STRING)
unset(PYTHON_DEBUG_LIBRARY CACHE)
unset(PYTHON_LIBRARY)
unset(PYTHON_LIBRARY_DEBUG)
unset(PYTHON_LIBRARY_RELEASE)
unset(PYTHON_LIBRARY CACHE)
unset(PYTHON_LIBRARY_DEBUG CACHE)
unset(PYTHON_LIBRARY_RELEASE CACHE)
unset(PYTHON_INCLUDE_DIR CACHE)
unset(PYTHON_INCLUDE_DIR2 CACHE)

# Locate requested python version
message(STATUS "jetson-utils: detecting Python v${PYTHON_BINDING_VERSION}...")

find_package(PythonInterp ${PYTHON_BINDING_VERSION} QUIET)
find_package(PythonLibs ${PYTHON_BINDING_VERSION} QUIET)

if(NOT ${PYTHONLIBS_FOUND})
    message(WARNING "jetson-utils: Python v${PYTHON_BINDING_VERSION} NOT found")
    return()
endif()

message(STATUS "jetson-utils: found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
message(STATUS "jetson-utils: found Python include:  ${PYTHON_INCLUDE_DIRS}")
message(STATUS "jetson-utils: found Python library:  ${PYTHON_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_DIRS})

# Attempt to find numpy
find_package(NumPy QUIET)

if(${NUMPY_FOUND})
  message(STATUS "jetson-utils: found NumPy version:  ${NUMPY_VERSION}")
  message(STATUS "jetson-utils: found NumPy include:  ${NUMPY_INCLUDE_DIR}")

  add_definitions(-DHAS_NUMPY)
  include_directories(${NUMPY_INCLUDE_DIR})
  link_directories(${NUMPY_INCLUDE_DIR}/../lib)
else()
  message(WARNING "jetson-utils: NumPy NOT found")
endif()

# Find source and header files
file(GLOB pythonUtilsSources *.cpp)
file(GLOB pythonUtilsIncludes *.h)

# Build Python bindings
cuda_add_library(${PROJECT_NAME}-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} SHARED ${pythonUtilsSources} ${pythonUtilsIncludes})

if(HAVE_WRITE_STRINGS)
  target_compile_options(${PROJECT_NAME}-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} PRIVATE -Wno-write-strings)
endif()

target_link_libraries(${PROJECT_NAME}-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
  jetson-utils
  ${PYTHON_LIBRARIES}
)

if(${NUMPY_FOUND})
  target_link_libraries(${PROJECT_NAME}-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
    npymath
  )
endif()

set_target_properties(${PROJECT_NAME}-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
  PROPERTIES
    PREFIX ""
    OUTPUT_NAME "jetson_utils_python"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
)

# on x86, install under /opt/conda/lib/pythonX.X/site-packages
# otherwise, install under /usr/local/lib/pythonX.X/dist-packages
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(PYTHON_BINDING_INSTALL_DIR /opt/conda/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
else()
  set(PYTHON_BINDING_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages)
endif()

# Install the bindings as a Python package
install(TARGETS ${PROJECT_NAME}-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
  DESTINATION ${PYTHON_BINDING_INSTALL_DIR}
)

# Install Python packaging files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/../module/__init__.py
  DESTINATION ${PYTHON_BINDING_INSTALL_DIR}/Jetson
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/../module/deprecated/__init__.py
  DESTINATION ${PYTHON_BINDING_INSTALL_DIR}/Jetson/Utils
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/../module/__init__.py
  DESTINATION ${PYTHON_BINDING_INSTALL_DIR}/jetson
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/../module/deprecated/__init__.py
  DESTINATION ${PYTHON_BINDING_INSTALL_DIR}/jetson/utils
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/../module/utils/__init__.py
  DESTINATION ${PYTHON_BINDING_INSTALL_DIR}/jetson_utils
)
