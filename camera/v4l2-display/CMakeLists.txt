# jetson-utils::v4l2-display -- CMakeLists.txt
cmake_minimum_required(VERSION 2.8)

# CMP0048: The project() command manages VERSION variables
cmake_policy(SET CMP0048 NEW)

# CMP0028: Double colon in target name means ALIAS or IMPORTED target
cmake_policy(SET CMP0028 NEW)

# Set name of the project
project(v4l2-display LANGUAGES CXX VERSION 1.0.0)

# Find source and header files
file(GLOB v4l2DisplaySources *.cpp)
file(GLOB v4l2DisplayIncludes *.h )

# Create the executable
add_executable(${PROJECT_NAME} ${v4l2DisplaySources} ${v4l2DisplayIncludes})
target_link_libraries(${PROJECT_NAME}
  jetson-utils
)

# Install the executable
install(TARGETS ${PROJECT_NAME}
  DESTINATION
    ${CMAKE_INSTALL_BINDIR}
)
