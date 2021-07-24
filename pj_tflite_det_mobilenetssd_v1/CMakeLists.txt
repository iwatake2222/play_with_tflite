cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "main")
project(${ProjectName})

# Select build system and set compile options
include(${CMAKE_CURRENT_LIST_DIR}/../InferenceHelper/common_helper/cmakes/build_setting.cmake)

# Create executable file
add_executable(${ProjectName} main.cpp)

# Link ImageProcessor module
add_subdirectory(./image_processor image_processor)
target_include_directories(${ProjectName} PUBLIC ./image_processor)
target_link_libraries(${ProjectName} ImageProcessor)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

# Copy resouce
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_definitions(-DRESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")

set(SPEED_TEST_ONLY on CACHE BOOL "Speed test only (don't use camera)? [on/off]")
if(SPEED_TEST_ONLY)
    add_definitions(-DSPEED_TEST_ONLY)
endif()
