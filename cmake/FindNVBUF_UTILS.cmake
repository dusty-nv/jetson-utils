# FindNVBUF_UTILS.cmake

find_library(NVBUF_UTILS_LIBRARY
  NAMES nvbuf_utils
  PATHS /usr/lib/aarch64-linux-gnu/tegra
)

find_path(
  NVBUF_UTILS_INCLUDE_DIR
  NAMES include/nvbuf_utils.h
  PATHS /usr/src/jetson_multimedia_api
)

if(NVBUF_UTILS_FOUND)
  mark_as_advanced(NVBUF_UTILS_INCLUDE_DIR)
  mark_as_advanced(NVBUF_UTILS_LIBRARY)
endif()

if(NVBUF_UTILS_FOUND AND NOT TARGET JetsonUtils::NVBUF_UTILS)
  add_library(JetsonUtils::NVBUF_UTILS IMPORTED)
  set_property(TARGET JetsonUtils::NVBUF_UTILS PROPERTY IMPORTED_LOCATION ${NVBUF_UTILS_LIBRARY})
  target_include_directories(JetsonUtils::NVBUF_UTILS INTERFACE ${NVBUF_UTILS_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVBUF_UTILS
  REQUIRED_VARS
    NVBUF_UTILS_LIBRARY NVBUF_UTILS_INCLUDE_DIR
)
