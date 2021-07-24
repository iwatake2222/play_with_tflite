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
#include "inference_helper.h"


class ClassificationEngine {
public:
	enum {
		kRetOk = 0,
		kRetErr = -1,
	};

	typedef struct Result_ {
		int32_t     class_id;
		std::string class_name;
		float     score;
		double    time_pre_process;		// [msec]
		double    time_inference;		// [msec]
		double    time_post_process;	// [msec]
		Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} Result;

private:
	static constexpr bool with_background_ = false;

public:
	ClassificationEngine() {}
	~ClassificationEngine() {}
	int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t Finalize(void);
	int32_t Process(const cv::Mat& original_mat, Result& result);

private:
	int32_t ReadLabel(const std::string& filename, std::vector<std::string>& label_list);

private:
	std::unique_ptr<InferenceHelper> inference_helper_;
	std::vector<InputTensorInfo> input_tensor_info_list_;
	std::vector<OutputTensorInfo> output_tensor_info_list_;
	std::vector<std::string> label_list_;
};

#endif
