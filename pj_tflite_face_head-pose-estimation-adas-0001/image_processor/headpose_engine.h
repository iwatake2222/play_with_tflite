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
#ifndef HEADPOSE_ENGINE_
#define HEADPOSE_ENGINE_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "inference_helper.h"
#include "bounding_box.h"

class HeadposeEngine {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef struct Result_ {
        float  yaw;             // Tait-Bryan angles [deg]
        float  pitch;           // Tait-Bryan angles [deg]
        float  roll;            // Tait-Bryan angles [deg]
        double time_pre_process;    // [msec]
        double time_inference;      // [msec]
        double time_post_process;   // [msec]
        Result_() : yaw(0), pitch(0), roll(0), time_pre_process(0), time_inference(0), time_post_process(0)
        {}
    } Result;

public:
    HeadposeEngine() {}
    ~HeadposeEngine() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, const std::vector<BoundingBox>& bbox_list, std::vector<Result>& result_list);


private:
    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;
};

#endif
