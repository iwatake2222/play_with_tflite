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
		RET_OK = 0,
		RET_ERR = -1,
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

	typedef struct RESULT_ {
		HAND_LANDMARK  handLandmark;
		double       timePreProcess;		// [msec]
		double       timeInference;		// [msec]
		double       timePostProcess;		// [msec]
		RESULT_() : timePreProcess(0), timeInference(0), timePostProcess(0)
		{}
	} RESULT;

public:
	HandLandmarkEngine() {}
	~HandLandmarkEngine() {}
	int32_t initialize(const std::string& workDir, const int32_t numThreads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, int32_t palmX, int32_t palmY, int32_t palmW, int32_t palmH, float palmRotation, RESULT& result);

public:
	void rotateLandmark(HAND_LANDMARK& handLandmark, float rotationRad, int32_t imageWidth, int32_t imageHeight);
	float calculateRotation(const HAND_LANDMARK& handLandmark);
	void transformLandmarkToRect(HAND_LANDMARK& handLandmark);

private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
};

#endif
