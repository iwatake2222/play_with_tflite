
#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

namespace cv {
	class Mat;
};


typedef struct {
	char labelFilename[256];
	int  numThreads;
} INPUT_PARAM;

typedef struct {
	int classId;
	char label[256];
	double score;
} OUTPUT_PARAM;

int ImageProcessor_initialize(const char *modelFilename, INPUT_PARAM *inputParam);
int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam);
int ImageProcessor_finalize(void);

#endif
