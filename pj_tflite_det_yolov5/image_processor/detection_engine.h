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
#ifndef DETECTION_ENGINE_
#define DETECTION_ENGINE_

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


class DetectionEngine {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef struct Object_ {
        int32_t     class_id;
        std::string label;
        float       score;
        int32_t     x;
        int32_t     y;
        int32_t     width;
        int32_t     height;
        Object_() : class_id(0), label(""), score(0), x(0), y(0), width(0), height(0)
        {}
    } Object;

    typedef struct Result_ {
        std::vector<Object> object_list;
        double              time_pre_process;		// [msec]
        double              time_inference;		// [msec]
        double              time_post_process;	// [msec]
        Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
        {}
    } Result;

public:
    DetectionEngine() {}
    ~DetectionEngine() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);

private:
    int32_t ReadLabel(const std::string& filename, std::vector<std::string>& label_list);
    float CalculateIoU(const Object& obj0, const Object& obj1);
    void Nms(std::vector<Object>& object_list, std::vector<Object>& object_nms_list);
    

private:
    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;
    std::vector<std::string> label_list_;
};

#endif
