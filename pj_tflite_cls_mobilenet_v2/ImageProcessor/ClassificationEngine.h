#ifndef CLASSIFICATION_ENGINE_
#define CLASSIFICATION_ENGINE_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "InferenceHelper.h"


class ClassificationEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef struct {
		int32_t     labelIndex;
		std::string labelName;
		float_t     score;
		double_t    timePreProcess;		// [msec]
		double_t    timeInference;		// [msec]
		double_t    timePostProcess;	// [msec]
	} RESULT;

public:
	ClassificationEngine() {}
	~ClassificationEngine() {}
	int32_t initialize(const std::string& workDir, const int32_t numThreads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);

private:
	int32_t readLabel(const std::string& filename, std::vector<std::string>& labelList);

private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
	std::vector<std::string> m_labelList;
};

#endif
