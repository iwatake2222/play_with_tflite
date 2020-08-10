
#ifndef CLASSIFY_
#define CLASSIFY_

#include <opencv2/opencv.hpp>
#include "InferenceHelper.h"

class Classify {
public:
	typedef struct {
		int labelIndex;
		std::string labelName;
		float score;
	} RESULT;

public:
	Classify() {}
	~Classify() {}
	int initialize(const char *workDir, const int numThreads);
	int finalize(void);
	int invoke(cv::Mat &originalMat, RESULT &result);

private:
	void readLabel(const std::string filename, std::vector<std::string> & labels);

private:
	InferenceHelper *m_inferenceHelper;
	TensorInfo *m_inputTensor;
	TensorInfo *m_outputTensor;
	std::vector<std::string> m_labels;
};

#endif
