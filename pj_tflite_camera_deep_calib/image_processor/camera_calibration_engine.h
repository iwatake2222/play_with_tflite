/* Copyright 2020 iwatake2222

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
#ifndef CAMERA_CALIBRATION_ENGINE_
#define CAMERA_CALIBRATION_ENGINE_

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


class CameraCalibrationEngine {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef struct Result_ {
        float     xi;
        float     focal_length;
        double    time_pre_process;		// [msec]
        double    time_inference;		// [msec]
        double    time_post_process;	// [msec]
        Result_() : xi(0), focal_length(0), time_pre_process(0), time_inference(0), time_post_process(0)
        {}
    } Result;

public:
    CameraCalibrationEngine() {}
    ~CameraCalibrationEngine() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);

private:
    int32_t GetMaxIndex(std::vector<float> value_list);

private:
    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;

    std::vector<float> class_dist_list_;
    std::vector<float> class_focal_list_;

    
};

#endif
