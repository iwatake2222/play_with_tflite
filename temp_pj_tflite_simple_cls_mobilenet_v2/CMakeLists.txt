cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "main")
project(${ProjectName})

# Switch build system
set(BUILD_SYSTEM auto CACHE STRING "Build target? [auto, x64_windows, x64_linux, armv7, aarch64]")
if(${BUILD_SYSTEM} STREQUAL "auto")
	if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
		set(BUILD_SYSTEM armv7)
	elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
		set(BUILD_SYSTEM aarch64)
	else()
		if(WIN32)
			set(BUILD_SYSTEM x64_windows)
		else()
			set(BUILD_SYSTEM x64_linux)
		endif()
	endif()
endif()
message("[main] CMAKE_SYSTEM_PROCESSOR = " ${CMAKE_SYSTEM_PROCESSOR} ", BUILD_SYSTEM = " ${BUILD_SYSTEM})

# Compile option
if(NOT MSVC)
	set(CMAKE_C_FLAGS "-Wall")
	set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
	set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
	set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -lstdc++")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
	if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
		message(STATUS "No build type selected, default to Release")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
	endif()
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Create executable file
add_executable(${ProjectName} Main.cpp)

# Link ImageProcessor module
add_subdirectory(./ImageProcessor ImageProcessor)
target_include_directories(${ProjectName} PUBLIC ./ImageProcessor)
target_link_libraries(${ProjectName} ImageProcessor)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})


# Copy resouce
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_definitions(-DRESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")

