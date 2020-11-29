#ifndef INFERENCE_HELPER_NCNN_
#define INFERENCE_HELPER_NCNN_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for ncnn */
#include "net.h"

/* for My modules */
#include "InferenceHelper.h"

class InferenceHelperNcnn : public InferenceHelper {
public:
	InferenceHelperNcnn();
	~InferenceHelperNcnn() override;
	int32_t setNumThread(const int32_t numThread) override;
	int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
	int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
	int32_t finalize(void) override;
	int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
	int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
	std::unique_ptr<ncnn::Net> m_net;
	std::vector<std::pair<std::string, ncnn::Mat>> m_inMatList;	// <name, mat>
	std::vector<ncnn::Mat> m_outMatList;
	int32_t m_numThread;
};

#endif
