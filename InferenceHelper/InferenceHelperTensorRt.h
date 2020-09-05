
#ifndef INFERENCE_HELPER_TENSORRT_
#define INFERENCE_HELPER_TENSORRT_

#include "InferenceHelper.h"

namespace nvinfer1 {
	class IRuntime;
	class ICudaEngine;
	class IExecutionContext;
}
class Logger;

class InferenceHelperTensorRt : public InferenceHelper {
public:
	InferenceHelperTensorRt();
	~InferenceHelperTensorRt() override {};
	int initialize(const char *modelFilename, int numThreads) override;
	int initialize(const char *modelFilename, const int numThreads, std::vector<std::pair<const char*, const void*>> customOps) override;
	int finalize(void) override;
	int invoke(void) override;
	int getTensorByName(const char *name, TensorInfo *tensorInfo) override;
	int getTensorByIndex(const int index, TensorInfo *tensorInfo) override;
	int setBufferToTensorByName(const char *name, void *data, const int dataSize) override;
	int setBufferToTensorByIndex(const int index, void *data, const int dataSize) override;

private:
	void allocateBuffers();

private:
	Logger* m_logger;
	nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;
	std::vector<std::pair<void*, int>> m_bufferListCPU;			// pointer and size (can be overwritten by user)
	std::vector<std::pair<void*, int>> m_bufferListCPUReserved;	// pointer and size (fixed in initialization)
	std::vector<void*> m_bufferListGPU;
};

#endif
