cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} ImageProcessor.cpp ImageProcessor.h)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

# For Tensorflow Lite
include(${CMAKE_CURRENT_LIST_DIR}/../../third_party/cmakes/tensorflow.cmake)
target_include_directories(${LibraryName} PUBLIC ${TFLITE_INC})
target_link_libraries(${LibraryName} ${TFLITE_LIB})
