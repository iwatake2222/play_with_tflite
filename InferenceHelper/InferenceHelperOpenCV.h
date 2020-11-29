#ifndef INFERENCE_HELPER_OPENCV_
#define INFERENCE_HELPER_OPENCV_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/* for My modules */
#include "InferenceHelper.h"

class InferenceHelperOpenCV : public InferenceHelper {
public:
	InferenceHelperOpenCV();
	~InferenceHelperOpenCV() override;
	int32_t setNumThread(const int32_t numThread) override;
	int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
	int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
	int32_t finalize(void) override;
	int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
	int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
	cv::dnn::Net m_net;
	std::vector<cv::Mat> m_inMatList;
	std::vector<cv::Mat> m_outMatList;	// store data as member variable so that an user can refer the results
};

#endif
