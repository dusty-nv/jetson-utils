
file(GLOB glDisplayTestSources *.cpp *.cu)
file(GLOB glDisplayTestIncludes *.h )

cuda_add_executable(gl-display-test ${glDisplayTestSources})
target_link_libraries(gl-display-test jetson-utils)

install(TARGETS gl-display-test DESTINATION bin)
