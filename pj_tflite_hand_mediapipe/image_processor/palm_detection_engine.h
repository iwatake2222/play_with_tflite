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
#include "inference_helper.h"


class PalmDetectionEngine {
public:
	enum {
		kRetOk = 0,
		kRetErr = -1,
	};

	typedef struct {
		float score;
		float rotation;
		float x;			// coordinate on the input image
		float y;
		float width;
		float height;
	} PALM;

	typedef struct Result_ {
		std::vector<PALM> palmList;
		double          time_pre_process;		// [msec]
		double          time_inference;		// [msec]
		double          time_post_process;		// [msec]
		Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} Result;

public:
	PalmDetectionEngine() {}
	~PalmDetectionEngine() {}
	int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t Finalize(void);
	int32_t Process(const cv::Mat& original_mat, Result& result);

private:
	std::unique_ptr<InferenceHelper> inference_helper_;
	std::vector<InputTensorInfo> input_tensor_info_list_;
	std::vector<OutputTensorInfo> output_tensor_info_list_;
};

#endif
