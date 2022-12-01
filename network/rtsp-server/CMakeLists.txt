
file(GLOB rtspServerSources *.cpp)
file(GLOB rtspServerIncludes *.h )

add_executable(rtsp-server ${rtspServerSources})
target_link_libraries(rtsp-server jetson-utils)

install(TARGETS rtsp-server DESTINATION bin)
