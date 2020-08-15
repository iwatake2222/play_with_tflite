
#ifndef STYLE_TRANSFER_H
#define STYLE_TRANSFER_H

#include <opencv2/opencv.hpp>
#include "InferenceHelper.h"

class StyleTransfer {
public:
	typedef struct {
		float *result;		// data is gauranteed until next invoke or calling destructor
	} STYLE_TRANSFER_RESULT;
public:
	StyleTransfer() {}
	~StyleTransfer() {}
	int initialize(const char *workDir, const int numThreads);
	int finalize(void);
	int invoke(cv::Mat &originalMat, const float styleBottleneck[], const int lengthStyleBottleneck, STYLE_TRANSFER_RESULT& result);

private:
	InferenceHelper *m_inferenceHelper;
	TensorInfo *m_inputTensor;
	TensorInfo *m_inputTensorStyleBottleneck;
	TensorInfo *m_outputTensor;
};

#endif
