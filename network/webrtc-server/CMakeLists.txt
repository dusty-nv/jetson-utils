
file(GLOB WebRTCServerSources *.cpp)
file(GLOB WebRTCServerIncludes *.h )

add_executable(webrtc-server ${WebRTCServerSources})
target_link_libraries(webrtc-server jetson-utils)

install(TARGETS webrtc-server DESTINATION bin)
