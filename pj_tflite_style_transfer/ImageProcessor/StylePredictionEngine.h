#ifndef STYLE_PREDICTION_ENGINE_
#define STYLE_PREDICTION_ENGINE_

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


class StylePredictionEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	static constexpr int SIZE_STYLE_BOTTLENECK = 100;

	typedef struct RESULT_ {
		const float_t*      styleBottleneck;	// data is gauranteed until next invoke or calling destructor
		double_t            timePreProcess;		// [msec]
		double_t            timeInference;		// [msec]
		double_t            timePostProcess;	// [msec]
		RESULT_() : timePreProcess(0), timeInference(0), timePostProcess(0)
		{}
	} RESULT;

public:
	StylePredictionEngine() {}
	~StylePredictionEngine() {}
	int32_t initialize(const std::string& workDir, const int32_t numThreads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);


private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
};

#endif
