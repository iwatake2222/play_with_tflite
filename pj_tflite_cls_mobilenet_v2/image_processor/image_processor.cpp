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
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "classification_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<ClassificationEngine> s_classification_engine;

/*** Function ***/
static cv::Scalar CreateCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}


int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam* input_param)
{
    if (s_classification_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_classification_engine.reset(new ClassificationEngine());
    if (s_classification_engine->Initialize(input_param->work_dir, input_param->num_threads) != ClassificationEngine::kRetOk) {
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_classification_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_classification_engine->Finalize() != ClassificationEngine::kRetOk) {
        return -1;
    }

    s_classification_engine.reset();

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_classification_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return -1;
    }
}


int32_t ImageProcessor::Process(cv::Mat* mat, ImageProcessor::OutputParam* output_param)
{
    if (!s_classification_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    cv::Mat& original_mat = *mat;
    ClassificationEngine::Result result;
    if (s_classification_engine->Process(original_mat, result) != ClassificationEngine::kRetOk) {
        return -1;
    }

    /* Draw the result */
    std::string result_str;
    result_str = "Result:" + result.class_name + " (score = " + std::to_string(result.score) + ")";
    cv::putText(original_mat, result_str, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, CreateCvColor(0, 0, 0), 3);
    cv::putText(original_mat, result_str, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, CreateCvColor(0, 255, 0), 1);

    /* Return the results */
    output_param->class_id = result.class_id;
    snprintf(output_param->label, sizeof(output_param->label), "%s", result.class_name.c_str());
    output_param->score = result.score;
    output_param->time_pre_process = result.time_pre_process;
    output_param->time_inference = result.time_inference;
    output_param->time_post_process = result.time_post_process;

    return 0;
}

