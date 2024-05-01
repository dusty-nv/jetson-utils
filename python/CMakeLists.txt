
#
# This list contains the versions of Python that will be detected 
# and the bindings built against (if that version is installed).
#
# If you want to support another version of Python, add it here.
#
if(USE_CONDA AND DEFINED ENV{CONDA_PREFIX})
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    if(PYTHON_VERSION_STRING)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" PYTHON_BINDING_VERSIONS "${PYTHON_VERSION_STRING}")
    else()
        set(PYTHON_BINDING_VERSIONS 3.9 3.10 3.11) # Default versions if specific version not detected
    endif()
else()
    if (LSB_RELEASE_CODENAME MATCHES "noble")
        set(PYTHON_BINDING_VERSIONS 3.12)
    elseif(LSB_RELEASE_CODENAME MATCHES "jammy")
        set(PYTHON_BINDING_VERSIONS 3.12)
    elseif(LSB_RELEASE_CODENAME MATCHES "focal")
        set(PYTHON_BINDING_VERSIONS 3.8)
    else()
        set(PYTHON_BINDING_VERSIONS 2.7 3.6 3.7)
    endif()
endif()

message("-- trying to build Python bindings for Python versions:  ${PYTHON_BINDING_VERSIONS}")

foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
    add_subdirectory(bindings bindings_python_${PYTHON_BINDING_VERSION})
endforeach()


#
# install Python examples
#
file(GLOB pyUtilExamples examples/*.py examples/*.sh)

foreach(pyExample ${pyUtilExamples})
    message("-- Copying ${pyExample}")
    file(COPY ${pyExample} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    install(FILES "${pyExample}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endforeach()