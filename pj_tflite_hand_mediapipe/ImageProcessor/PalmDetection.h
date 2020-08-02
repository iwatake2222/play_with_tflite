
#ifndef PALM_DETECTOR_
#define PALM_DETECTOR_

#include <opencv2/opencv.hpp>
#include "InferenceHelper.h"

class PalmDetection {
public:
	typedef struct {
		float score;
		float rotation;
		float x;			// coordinate on the input image
		float y;
		float width;
		float height;
	} PALM;

public:
	PalmDetection() {}
	~PalmDetection() {}
	int initialize(const char *workDir, const int numThreads);
	int finalize(void);
	int invoke(cv::Mat &originalMat, std::vector<PALM> &palmList);

private:

private:
	InferenceHelper *m_inferenceHelper;
	TensorInfo *m_inputTensor;
	TensorInfo *m_outputTensorBoxes;
	TensorInfo *m_outputTensorScores;
};

#endif
