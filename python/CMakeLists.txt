# jetson-utils::python -- CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

# CMP0048: The project() command manages VERSION variables
cmake_policy(SET CMP0048 NEW)

# CMP0028: Double colon in target name means ALIAS or IMPORTED target
cmake_policy(SET CMP0028 NEW)

# Set name of the project
project(jetson-utils-python LANGUAGES CXX VERSION 1.0.0)

# Find Python versions or use the user-defined version
if(NOT DEFINED JU_PYTHON_BINDING_VERSION)
  if(DEFINED LSB_RELEASE_CODENAME)
    if(LSB_RELEASE_CODENAME MATCHES "focal")
      set(PYTHON_BINDING_VERSIONS 3.8)
    else()
      set(PYTHON_BINDING_VERSIONS 2.7 3.6 3.7)
    endif()
  endif()

  if(NOT DEFINED PYTHON_BINDING_VERSIONS)
    message(WARNING "jetson-utils: LSB_RELEASE_CODENAME is not defined. Identifying Python versions from OS default CMake version")
    if(CMAKE_VERSION VERSION_GREATER "3.10.3")
      set(PYTHON_BINDING_VERSIONS 3.8)
    else()
      set(PYTHON_BINDING_VERSIONS 2.7 3.6 3.7)
    endif()
  endif()
else()
  set(PYTHON_BINDING_VERSIONS ${JU_PYTHON_BINDING_VERSION})
endif()

# Build Python bindings
message(STATUS "jetson-utils: Building jetson-utils bindings for Python versions:  ${PYTHON_BINDING_VERSIONS}")
foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
    add_subdirectory(bindings ${CMAKE_CURRENT_BINARY_DIR}/bindings_python_${PYTHON_BINDING_VERSION})
endforeach()

# Install Python examples
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/examples
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/examples
)

# Configure Python module __init__.py files
configure_file(${CMAKE_CURRENT_LIST_DIR}/init_empty.py.in ${CMAKE_CURRENT_BINARY_DIR}/module/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/init_deprecated.py.in ${CMAKE_CURRENT_BINARY_DIR}/module/deprecated/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/init.py.in ${CMAKE_CURRENT_BINARY_DIR}/module/utils/__init__.py COPYONLY)
