# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_simple_trajectory_generator_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED simple_trajectory_generator_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(simple_trajectory_generator_FOUND FALSE)
  elseif(NOT simple_trajectory_generator_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(simple_trajectory_generator_FOUND FALSE)
  endif()
  return()
endif()
set(_simple_trajectory_generator_CONFIG_INCLUDED TRUE)

# output package information
if(NOT simple_trajectory_generator_FIND_QUIETLY)
  message(STATUS "Found simple_trajectory_generator: 0.0.0 (${simple_trajectory_generator_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'simple_trajectory_generator' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${simple_trajectory_generator_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(simple_trajectory_generator_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_dependencies-extras.cmake")
foreach(_extra ${_extras})
  include("${simple_trajectory_generator_DIR}/${_extra}")
endforeach()
