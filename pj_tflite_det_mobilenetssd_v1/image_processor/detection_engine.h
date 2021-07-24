#ifndef DETECTION_ENGINE_
#define DETECTION_ENGINE_

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


class DetectionEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef struct {
		int32_t     class_id;
		std::string label;
		float  score;
		float  x;
		float  y;
		float  width;
		float  height;
	} OBJECT;

	typedef struct RESULT_ {
		std::vector<OBJECT> object_list;
		double            time_pre_process;		// [msec]
		double            time_inference;		// [msec]
		double            time_post_process;	// [msec]
		RESULT_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} RESULT;

public:
	DetectionEngine() {}
	~DetectionEngine() {}
	int32_t initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);

private:
	int32_t readLabel(const std::string& filename, std::vector<std::string>& labelList);
	int32_t getObject(std::vector<OBJECT>& object_list, const float *outputBoxList, const float *outputClassList, const float *outputScoreList, const int32_t outputNum,
		const double threshold, const int32_t width, const int32_t height);

private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
	std::vector<std::string> m_labelList;
};

#endif
