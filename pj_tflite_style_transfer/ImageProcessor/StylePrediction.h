
#ifndef STYLE_PREDICTION_H
#define STYLE_PREDICTION_H

#include <opencv2/opencv.hpp>
#include "InferenceHelper.h"

class StylePrediction {
public:
	static const int SIZE_STYLE_BOTTLENECK = 100;
	typedef struct {
		float *styleBottleneck;		// data is gauranteed until next invoke or calling destructor
	} STYLE_PREDICTION_RESULT;

public:
	StylePrediction() {}
	~StylePrediction() {}
	int initialize(const char *workDir, const int numThreads);
	int finalize(void);
	int invoke(cv::Mat &originalMat, STYLE_PREDICTION_RESULT& result);

private:
	InferenceHelper *m_inferenceHelper;
	TensorInfo *m_inputTensor;
	TensorInfo *m_outputTensor;
};

#endif
