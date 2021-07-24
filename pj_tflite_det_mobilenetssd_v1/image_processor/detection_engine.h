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

	typedef struct Object_ {
		int32_t     class_id;
		std::string label;
		float  score;
		float  x;
		float  y;
		float  width;
		float  height;
		Object_() : class_id(0), label(""), score(0), x(0), y(0), width(0), height(0)
		{}
	} Object;

	typedef struct Result_ {
		std::vector<Object> object_list;
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
	int32_t ReadLabel(const std::string& filename, std::vector<std::string>& label_list);
	int32_t GetObject(std::vector<Object>& object_list, const float *output_box_list, const float *output_class_list, const float *output_score_list, const int32_t output_num,
		const double threshold, const int32_t width, const int32_t height);

private:
	std::unique_ptr<InferenceHelper> inference_helper_;
	std::vector<InputTensorInfo> input_tensor_info_list_;
	std::vector<OutputTensorInfo> output_tensor_info_list_;
	std::vector<std::string> label_list_;
};

#endif
