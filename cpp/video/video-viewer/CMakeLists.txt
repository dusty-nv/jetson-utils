
file(GLOB videoViewerSources *.cpp)
file(GLOB videoViewerIncludes *.h )

add_executable(video-viewer ${videoViewerSources})
target_link_libraries(video-viewer jetson-utils)

install(TARGETS video-viewer DESTINATION bin)
