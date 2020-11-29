cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} ImageProcessor.cpp ImageProcessor.h DetectionEngine.cpp DetectionEngine.h)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

# Link Common Helper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../CommonHelper CommonHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../CommonHelper)
target_link_libraries(${LibraryName} CommonHelper)

# Link InferenceHelper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper InferenceHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper)
target_link_libraries(${LibraryName} InferenceHelper)
