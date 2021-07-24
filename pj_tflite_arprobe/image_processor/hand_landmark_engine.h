#ifndef HAND_LANDMARK_ENGINE_
#define HAND_LANDMARK_ENGINE_

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


class HandLandmarkEngine {
public:
	enum {
		kRetOk = 0,
		kRetErr = -1,
	};

	typedef struct {
		float handflag;
		float handedness;
		struct {
			float x;	// coordinate on the input image
			float y;	// coordinate on the input image
			float z;
		} pos[21];
		struct {
			float x;	// coordinate on the input image
			float y;
			float width;
			float height;
			float rotation;
		} rect;
	} HAND_LANDMARK;

	typedef struct Result_ {
		HAND_LANDMARK  hand_landmark;
		double       time_pre_process;		// [msec]
		double       time_inference;		// [msec]
		double       time_post_process;		// [msec]
		Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} Result;

public:
	HandLandmarkEngine() {}
	~HandLandmarkEngine() {}
	int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t Finalize(void);
	int32_t Process(const cv::Mat& original_mat, int32_t palmX, int32_t palmY, int32_t palmW, int32_t palmH, float palmRotation, Result& result);

public:
	void RotateLandmark(HAND_LANDMARK& hand_landmark, float rotationRad, int32_t image_width, int32_t image_height);
	float CalculateRotation(const HAND_LANDMARK& hand_landmark);
	void TransformLandmarkToRect(HAND_LANDMARK& hand_landmark);

private:
	std::unique_ptr<InferenceHelper> inference_helper_;
	std::vector<InputTensorInfo> input_tensor_info_list_;
	std::vector<OutputTensorInfo> output_tensor_info_list_;
};

#endif
