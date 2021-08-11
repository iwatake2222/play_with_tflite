/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
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
} Result;

int32_t Initialize(const InputParam& input_param);
int32_t Process(cv::Mat& mat, Result& result);
int32_t Finalize(void);
int32_t Command(int32_t cmd);

}

#endif
