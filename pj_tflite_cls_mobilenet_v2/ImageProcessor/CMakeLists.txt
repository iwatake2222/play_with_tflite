cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} ImageProcessor.cpp ImageProcessor.h)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

# Link InferenceHelper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper InferenceHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper)
target_link_libraries(${LibraryName} InferenceHelper)


if(TFLITE_DELEGATE_EDGETPU)
	add_definitions(-DTFLITE_DELEGATE_EDGETPU)
endif()
if(TFLITE_DELEGATE_GPU)
	add_definitions(-DTFLITE_DELEGATE_GPU)
endif()
if(TFLITE_DELEGATE_XNNPACK)
	add_definitions(-DTFLITE_DELEGATE_XNNPACK)
endif()
