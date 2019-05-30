
file(GLOB gstCameraSources *.cpp)
file(GLOB gstCameraIncludes *.h )

add_executable(camera-viewer ${gstCameraSources})
target_link_libraries(camera-viewer jetson-utils)

install(TARGETS camera-viewer DESTINATION bin)
