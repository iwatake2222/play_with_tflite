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
#include "common_helper_cv.h"
#include "classification_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<ClassificationEngine> s_classification_engine;

/*** Function ***/
static void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}

int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam& input_param)
{
    if (s_classification_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_classification_engine.reset(new ClassificationEngine());
    if (s_classification_engine->Initialize(input_param.work_dir, input_param.num_threads) != ClassificationEngine::kRetOk) {
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


int32_t ImageProcessor::Process(cv::Mat& mat, ImageProcessor::Result& result)
{
    if (!s_classification_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    ClassificationEngine::Result cls_result;
    if (s_classification_engine->Process(mat, cls_result) != ClassificationEngine::kRetOk) {
        return -1;
    }

    /* Draw the result */
    char text[64];
    snprintf(text, sizeof(text), "Result: %s (score = %.3f)",  cls_result.class_name.c_str(), cls_result.score);
    CommonHelper::DrawText(mat, text, cv::Point(0, 20), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    DrawFps(mat, cls_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.class_id = cls_result.class_id;
    snprintf(result.label, sizeof(result.label), "%s", cls_result.class_name.c_str());
    result.score = cls_result.score;
    result.time_pre_process = cls_result.time_pre_process;
    result.time_inference = cls_result.time_inference;
    result.time_post_process = cls_result.time_post_process;

    return 0;
}

