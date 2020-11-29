#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

namespace cv {
	class Mat;
};

#define NUM_MAX_RESULT 100

typedef struct {
	char     workDir[256];
	int32_t  numThreads;
} INPUT_PARAM;

typedef struct {
	int32_t objectNum;
	struct {
		int32_t  classId;
		char     label[256];
		double_t score;
		int32_t  x;
		int32_t  y;
		int32_t  width;
		int32_t  height;
	} objectList[NUM_MAX_RESULT];
	double_t timePreProcess;   // [msec]
	double_t timeInference;    // [msec]
	double_t timePostProcess;  // [msec]
} OUTPUT_PARAM;

int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam);
int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam);
int32_t ImageProcessor_finalize(void);
int32_t ImageProcessor_command(int32_t cmd);

#endif
