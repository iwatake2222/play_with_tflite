#ifndef SEMANTIC_SEGMENTATION_ENGINE_
#define SEMANTIC_SEGMENTATION_ENGINE_

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
#include "inference_helper.h"


class SemanticSegmentationEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef struct RESULT_ {
		cv::Mat             maskImage;
		double            time_pre_process;		// [msec]
		double            time_inference;		// [msec]
		double            time_post_process;	// [msec]
		RESULT_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} RESULT;

public:
	SemanticSegmentationEngine() {}
	~SemanticSegmentationEngine() {}
	int32_t initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);


private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
};

#endif
