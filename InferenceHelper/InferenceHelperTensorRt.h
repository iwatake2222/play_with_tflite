#ifndef INFERENCE_HELPER_TENSORRT_
#define INFERENCE_HELPER_TENSORRT_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for My modules */
#include "InferenceHelper.h"

namespace nvinfer1 {
	class IRuntime;
	class ICudaEngine;
	class IExecutionContext;
}

class InferenceHelperTensorRt : public InferenceHelper {
public:
	InferenceHelperTensorRt();
	~InferenceHelperTensorRt() override {};
	int32_t setNumThread(const int32_t numThread) override;
	int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
	int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
	int32_t finalize(void) override;
	int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
	int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
	int32_t allocateBuffers(std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList);

private:
	int32_t m_numThread;
	std::shared_ptr<nvinfer1::IRuntime> m_runtime;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
	std::shared_ptr<nvinfer1::IExecutionContext> m_context;
	std::vector<std::pair<void*, int32_t>> m_bufferListCPU;			// pointer and size (can be overwritten by user)
	std::vector<std::pair<void*, int32_t>> m_bufferListCPUReserved;	// pointer and size (fixed in initialization)
	std::vector<void*> m_bufferListGPU;
};

#endif
