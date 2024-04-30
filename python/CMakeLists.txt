
#
# This list contains the versions of Python that will be detected 
# and the bindings built against (if that version is installed).
#
# If you want to support another version of Python, add it here.
#

# Option to use Python from a Conda environment
option(USE_CONDA "Use Python from the activated Conda environment" OFF)

if(LSB_RELEASE_CODENAME MATCHES "noble")
	set(PYTHON_BINDING_VERSIONS 3.12)
elseif(LSB_RELEASE_CODENAME MATCHES "jammy")
	set(PYTHON_BINDING_VERSIONS 3.10)
elseif(LSB_RELEASE_CODENAME MATCHES "focal")
	set(PYTHON_BINDING_VERSIONS 3.8)
else()
	set(PYTHON_BINDING_VERSIONS 2.7 3.6 3.7)
endif()

message("-- trying to build Python bindings for Python versions:  ${PYTHON_BINDING_VERSIONS}")

# Check if user wants to use Conda and if a Conda environment is activated
if(USE_CONDA AND DEFINED ENV{CONDA_PREFIX})
    message("-- Using Python from Conda environment: $ENV{CONDA_PREFIX}")
    foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
        # Explicitly setting Python executable to the one in the Conda environment
        set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python${PYTHON_BINDING_VERSION}")
        find_package(PythonInterp ${PYTHON_BINDING_VERSION} REQUIRED)
        find_package(PythonLibs ${PYTHON_BINDING_VERSION} REQUIRED)
        message("-- Configuring bindings with Python ${PYTHON_BINDING_VERSION} from Conda.")
        add_subdirectory(bindings bindings_python_${PYTHON_BINDING_VERSION})
    endforeach()
else()
    message("-- No Conda environment detected or Conda option not selected, using system Python.")
    foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
        find_package(PythonInterp ${PYTHON_BINDING_VERSION} REQUIRED)
        find_package(PythonLibs ${PYTHON_BINDING_VERSION} REQUIRED)
        add_subdirectory(bindings bindings_python_${PYTHON_BINDING_VERSION})
    endforeach()
endif()


#
# install Python examples
#
file(GLOB pyUtilExamples examples/*.py examples/*.sh)

foreach(pyExample ${pyUtilExamples})
	message("-- Copying ${pyExample}")
	file(COPY ${pyExample} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	install(FILES "${pyExample}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endforeach()
