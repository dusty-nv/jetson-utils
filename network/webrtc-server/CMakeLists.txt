# jetson-utils::webrtc-server -- CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

# CMP0048: The project() command manages VERSION variables
cmake_policy(SET CMP0048 NEW)

# CMP0028: Double colon in target name means ALIAS or IMPORTED target
cmake_policy(SET CMP0028 NEW)

# Set name of the project
project(webrtc-server LANGUAGES CXX VERSION 1.0.0)

# Find source and header files
file(GLOB WebRTCServerSources *.cpp)
file(GLOB WebRTCServerIncludes *.h )

# Create the executable
add_executable(${PROJECT_NAME} ${WebRTCServerSources} ${WebRTCServerIncludes})
target_link_libraries(${PROJECT_NAME}
  jetson-utils
)
set_target_properties(${PROJECT_NAME}
  PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../:$ORIGIN/../lib"
)

# Install the executable
install(TARGETS ${PROJECT_NAME}
  DESTINATION
    ${CMAKE_INSTALL_BINDIR}
)
