cmake_minimum_required(VERSION 3.0)

set(LibraryName "InferenceHelper")

set(INFERENCE_HELPER_ENABLE_TENSORRT off CACHE BOOL "With TensorRT? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE on CACHE BOOL "With Tflite? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU off CACHE BOOL "With Tflite Delegate EdgeTPU? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU off CACHE BOOL "With Tflite Delegate GPU? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK off CACHE BOOL "With Tflite Delegate XNNPACK? [on/off]")

# Create library
set(SRC InferenceHelper.h InferenceHelper.cpp)

if(INFERENCE_HELPER_ENABLE_TENSORRT)
	set(SRC ${SRC} InferenceHelperTensorRt.h InferenceHelperTensorRt.cpp)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE)
	set(SRC ${SRC} InferenceHelperTensorflowLite.h InferenceHelperTensorflowLite.cpp)
endif()
add_library(${LibraryName} ${SRC})

# For TensorRT
if(INFERENCE_HELPER_ENABLE_TENSORRT)
	find_package(CUDA REQUIRED)
	target_link_libraries(${LibraryName}
		${CUDA_LIBRARIES}
		nvinfer
		nvonnxparser
		nvinfer_plugin
		cudnn
	)
	target_include_directories(${LibraryName} PUBLIC
		${CUDA_INCLUDE_DIRS}
	)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TENSORRT)
endif()

# For Tensorflow Lite
if(INFERENCE_HELPER_ENABLE_TFLITE)
	include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/tflite.cmake)
	target_include_directories(${LibraryName} PUBLIC ${TFLITE_INC})
	target_link_libraries(${LibraryName} ${TFLITE_LIB})
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE)
endif()

# For Tensorflow Lite Delegate(Edge TPU)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
	include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/tflite_edgetpu.cmake)
	target_include_directories(${LibraryName} PUBLIC ${TFLITE_EDGETPI_INC})
	target_link_libraries(${LibraryName} ${TFLITE_EDGETPI_LIB})
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
endif()

# For Tensorflow Lite Delegate(GPU)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
	find_package(OpenCL)
	if(OpenCL_Found)
		target_include_directories(${LibraryName} PUBLIC ${OpenCL_INCLUDE_DIRS})
		target_link_libraries(${LibraryName} ${OpenCL_LIBRARIES})
	endif()
	include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/tflite_gpu.cmake)
	target_include_directories(${LibraryName} PUBLIC ${TFLITE_GPU_INC})
	target_link_libraries(${LibraryName} ${TFLITE_GPU_LIB} EGL GLESv2)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
endif()

# For Tensorflow Lite Delegate(XNNPACK)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
endif()
