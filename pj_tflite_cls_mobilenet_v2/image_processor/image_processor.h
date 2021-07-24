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


namespace ImageProcessor
{

typedef struct {
    char     work_dir[256];
    int32_t  num_threads;
} InputParam;

typedef struct {
    int32_t class_id;
    char    label[256];
    double  score;
    double  time_pre_process;   // [msec]
    double  time_inference;    // [msec]
    double  time_post_process;  // [msec]
} OutputParam;

int32_t Initialize(const InputParam* input_param);
int32_t Process(cv::Mat* mat, OutputParam* output_param);
int32_t Finalize(void);
int32_t Command(int32_t cmd);

}

#endif
