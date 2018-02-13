
file(GLOB glDisplayTestSources *.cpp)
file(GLOB glDisplayTestIncludes *.h )

add_executable(gl-display-test ${glDisplayTestSources})
target_link_libraries(gl-display-test jetson-utils)
