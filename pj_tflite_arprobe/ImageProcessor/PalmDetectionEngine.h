#ifndef PALM_DETECTION_ENGINE_
#define PALM_DETECTION_ENGINE_

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


class PalmDetectionEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef struct {
		float score;
		float rotation;
		float x;			// coordinate on the input image
		float y;
		float width;
		float height;
	} PALM;

	typedef struct RESULT_ {
		std::vector<PALM> palmList;
		double          timePreProcess;		// [msec]
		double          timeInference;		// [msec]
		double          timePostProcess;		// [msec]
		RESULT_() : timePreProcess(0), timeInference(0), timePostProcess(0)
		{}
	} RESULT;

public:
	PalmDetectionEngine() {}
	~PalmDetectionEngine() {}
	int32_t initialize(const std::string& workDir, const int32_t numThreads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);

private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
};

#endif
