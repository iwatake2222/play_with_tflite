cmake_minimum_required(VERSION 3.0)

set(LibraryName "InferenceHelper")

set(INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV on CACHE BOOL "Enable PreProcess by OpenCV? [on/off]")
set(INFERENCE_HELPER_ENABLE_OPENCV on CACHE BOOL "With OpenCV? [on/off]")
set(INFERENCE_HELPER_ENABLE_TENSORRT off CACHE BOOL "With TensorRT? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE on CACHE BOOL "With Tflite? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU off CACHE BOOL "With Tflite Delegate EdgeTPU? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU off CACHE BOOL "With Tflite Delegate GPU? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK off CACHE BOOL "With Tflite Delegate XNNPACK? [on/off]")
set(INFERENCE_HELPER_ENABLE_NCNN off CACHE BOOL "With Ncnn? [on/off]")
set(INFERENCE_HELPER_ENABLE_MNN off CACHE BOOL "With Mnn? [on/off]")

# Create library
set(SRC InferenceHelper.h InferenceHelper.cpp)

if(INFERENCE_HELPER_ENABLE_OPENCV)
	set(SRC ${SRC} InferenceHelperOpenCV.h InferenceHelperOpenCV.cpp)
endif()
if(INFERENCE_HELPER_ENABLE_TENSORRT)
	set(SRC ${SRC} InferenceHelperTensorRt.h InferenceHelperTensorRt.cpp)
	set(SRC ${SRC} TensorRT/logger.cpp TensorRT/BatchStream.h TensorRT/common.h TensorRT/EntropyCalibrator.h TensorRT/logger.h TensorRT/logging.h)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE)
	set(SRC ${SRC} InferenceHelperTensorflowLite.h InferenceHelperTensorflowLite.cpp)
endif()
if(INFERENCE_HELPER_ENABLE_NCNN)
	set(SRC ${SRC} InferenceHelperNcnn.h InferenceHelperNcnn.cpp)
endif()
if(INFERENCE_HELPER_ENABLE_MNN)
	set(SRC ${SRC} InferenceHelperMnn.h InferenceHelperMnn.cpp)
endif()
add_library(${LibraryName} ${SRC})

target_include_directories(${LibraryName} PUBLIC ../CommonHelper)
target_link_libraries(${LibraryName} CommonHelper)

# For TensorInfo (Pre process calculation)
if(INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV)
	find_package(OpenCV REQUIRED)
	target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${LibraryName} ${OpenCV_LIBS})
	add_definitions(-DINFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV)
endif()

# For OpenCV
if(INFERENCE_HELPER_ENABLE_OPENCV)
	find_package(OpenCV REQUIRED)
	target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${LibraryName} ${OpenCV_LIBS})
	add_definitions(-DINFERENCE_HELPER_ENABLE_OPENCV)
endif()

# For TensorRT
if(INFERENCE_HELPER_ENABLE_TENSORRT)
	find_package(CUDA)
	if(CUDA_FOUND)
		target_link_libraries(${LibraryName}
			${CUDA_LIBRARIES}
			nvinfer
			nvonnxparser
			nvinfer_plugin
			cudnn
		)
		target_include_directories(${LibraryName} PUBLIC
			${CUDA_INCLUDE_DIRS}
			TensorRT
		)
		add_definitions(-DINFERENCE_HELPER_ENABLE_TENSORRT)
		message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
	else()
		message(WARNING, "Cannot find CUDA")
	endif()
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


# For NCNN
if(INFERENCE_HELPER_ENABLE_NCNN)
	include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/ncnn.cmake)
	target_include_directories(${LibraryName} PUBLIC ${NCNN_INC})
	target_link_libraries(${LibraryName} ${NCNN_LIB})
	add_definitions(-DINFERENCE_HELPER_ENABLE_NCNN)
endif()


# For MNN
if(INFERENCE_HELPER_ENABLE_MNN)
	include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/MNN.cmake)
	target_include_directories(${LibraryName} PUBLIC ${MNN_INC})
	target_link_libraries(${LibraryName} ${MNN_LIB})
	add_definitions(-DINFERENCE_HELPER_ENABLE_MNN)
endif()
