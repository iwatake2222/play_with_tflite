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
#ifndef AGEGENDER_ENGINE_
#define AGEGENDER_ENGINE_

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

class AgeGenderEngine {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    enum {
        kGenderFemale = 0,
        kGenderMale = 1,
        kGenderNotSure = 2,
    };

    typedef struct Result_ {
        int32_t age;
        int32_t gender;
        std::string gender_str;
        double time_pre_process;    // [msec]
        double time_inference;      // [msec]
        double time_post_process;   // [msec]
        Result_() : age(0), gender(kGenderNotSure), gender_str("NotSure"), time_pre_process(0), time_inference(0), time_post_process(0)
        {}
    } Result;

public:
    AgeGenderEngine(float threshold_fender = 0.7f) 
        : threshold_fender_(threshold_fender)
    {}
    ~AgeGenderEngine() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, const BoundingBox& bbox, Result& result);
    static const std::vector<std::pair<int32_t, int32_t>>& GetConnectionList();


private:
    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;

    float threshold_fender_;
};

#endif
