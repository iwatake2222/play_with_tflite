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

namespace ImageProcessor
{

typedef struct {
	char     work_dir[256];
	int32_t  num_threads;
} InputParam;

typedef struct {
	int32_t object_num;
	struct {
		int32_t  class_id;
		char     label[256];
		double   score;
		int32_t  x;
		int32_t  y;
		int32_t  width;
		int32_t  height;
	} object_list[NUM_MAX_RESULT];
	double time_pre_process;   // [msec]
	double time_inference;    // [msec]
	double time_post_process;  // [msec]
} OutputParam;

int32_t Initialize(const InputParam* input_param);
int32_t Process(cv::Mat* mat, OutputParam* output_param);
int32_t Finalize(void);
int32_t Command(int32_t cmd);

}

#endif
