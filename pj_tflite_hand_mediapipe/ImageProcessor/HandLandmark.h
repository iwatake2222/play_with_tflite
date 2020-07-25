
#ifndef HAND_LANDMARK_
#define HAND_LANDMARK_

#include <opencv2/opencv.hpp>
#include "InferenceHelper.h"

class HandLandmark {
public:
	typedef struct {
		float handflag;
		float handedness;
		struct {
			float x;
			float y;
			float z;
		} pos[21];
	} HAND_LANDMARK;

public:
	HandLandmark() {}
	~HandLandmark() {}
	int initialize(const char *workDir, const int numThreads);
	int finalize(void);
	int invoke(cv::Mat &originalMat, HAND_LANDMARK& handLandmark);
	int rotateLandmark(HAND_LANDMARK& handLandmark, float rotationRa, int imageWidth, int imageHeight);
	float calculateRotation(HAND_LANDMARK& handLandmark);

private:
	InferenceHelper *m_inferenceHelper;
	TensorInfo *m_inputTensor;
	TensorInfo *m_outputTensorLd21;
	TensorInfo *m_outputTensorHandflag;
	TensorInfo *m_outputTensorHandedness;
};

#endif
