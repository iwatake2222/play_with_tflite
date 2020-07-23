# Create sub library for Edge TPU Pipiline
set(TFLITE_EDGETPU_PIPELINE "TFLITE_EDGETPU_PIPELINE")
add_library(${TFLITE_EDGETPU_PIPELINE} SHARED
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/allocator.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/common.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/utils.cc
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/utils.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/pipelined_model_runner.cc
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/pipelined_model_runner.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/aligned_alloc.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/default_allocator.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/memory_pool_allocator.cc
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/memory_pool_allocator.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/segment_runner.cc
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/segment_runner.h
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal/thread_safe_queue.h
)

target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../edgetpu)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../edgetpu/libedgetpu)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline/internal)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../tensorflow)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../tensorflow/tensorflow/lite/tools/make/downloads/absl)

# libraries used in pipeline
## glog
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../glog glog)
target_link_libraries(${TFLITE_EDGETPU_PIPELINE} glog)
target_include_directories(${TFLITE_EDGETPU_PIPELINE} PUBLIC ${CMAKE_BINARY_DIR}/glog)
## absl
set(BUILD_TESTING OFF CACHE BOOL "disable BUILD_TESTING for absl" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../abseil-cpp absl)
set(ABSL_LIBS absl_synchronization absl_stacktrace absl_symbolize absl_demangle_internal absl_debugging_internal absl_dynamic_annotations absl_time absl_time_zone absl_graphcycles_internal absl_failure_signal_handler absl_malloc_internal absl_base absl_spinlock_wait)
target_link_libraries(${TFLITE_EDGETPU_PIPELINE} ${ABSL_LIBS})
if(NOT WIN32)
	target_link_libraries(${TFLITE_EDGETPU_PIPELINE} atomic)
endif()

set(TFLITE_EDGETPU_PIPELINE_INC
	${CMAKE_CURRENT_LIST_DIR}/../edgetpu/src/cpp/pipeline
)
set(TFLITE_EDGETPU_PIPELINE_LIB
	${TFLITE_EDGETPU_PIPELINE}
)
