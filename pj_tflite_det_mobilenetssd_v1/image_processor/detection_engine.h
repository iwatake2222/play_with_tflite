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
		kRetOk = 0,
		kRetErr = -1,
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

	typedef struct Result_ {
		std::vector<OBJECT> object_list;
		double            time_pre_process;		// [msec]
		double            time_inference;		// [msec]
		double            time_post_process;	// [msec]
		Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} Result;

public:
	DetectionEngine() {}
	~DetectionEngine() {}
	int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t Finalize(void);
	int32_t Process(const cv::Mat& original_mat, Result& result);

private:
	int32_t ReadLabel(const std::string& filename, std::vector<std::string>& labelList);
	int32_t getObject(std::vector<OBJECT>& object_list, const float *outputBoxList, const float *outputClassList, const float *outputScoreList, const int32_t outputNum,
		const double threshold, const int32_t width, const int32_t height);

private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
	std::vector<std::string> m_labelList;
};

#endif
