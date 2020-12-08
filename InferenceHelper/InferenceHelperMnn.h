#ifndef INFERENCE_HELPER_MNN_
#define INFERENCE_HELPER_MNN_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for MNN */
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>

/* for My modules */
#include "InferenceHelper.h"

class InferenceHelperMnn : public InferenceHelper {
public:
	InferenceHelperMnn();
	~InferenceHelperMnn() override;
	int32_t setNumThread(const int32_t numThread) override;
	int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
	int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
	int32_t finalize(void) override;
	int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
	int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
	void convertNormalizeParameters(InputTensorInfo& inputTensorInfo);

private:
	std::unique_ptr<MNN::Interpreter> m_net;
	MNN::Session* m_session;
	std::vector<std::unique_ptr<MNN::Tensor>> m_outMatList;
	int32_t m_numThread;
};

#endif
