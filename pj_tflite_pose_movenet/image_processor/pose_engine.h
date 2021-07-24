#ifndef POSE_ENGINE_
#define POSE_ENGINE_

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


class PoseEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef struct RESULT_ {
		std::vector<float>                                  poseScores;			// [body]
		std::vector<std::vector<float>>                     poseKeypointScores;	// [body][part]
		std::vector<std::vector<std::pair<float, float>>>   poseKeypointCoords;	// [body][part][x, y] (0 - 1.0)
		double    time_pre_process;		// [msec]
		double    time_inference;		// [msec]
		double    time_post_process;	// [msec]
		RESULT_() : time_pre_process(0), time_inference(0), time_post_process(0)
		{}
	} RESULT;

private:
	static constexpr bool WITH_BACKGROUND = true;

public:
	PoseEngine() {}
	~PoseEngine() {}
	int32_t initialize(const std::string& work_dir, const int32_t num_threads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);
private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
};

#endif
