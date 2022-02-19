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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "segmentation_engine.h"
#include "image_processor.h"

/*** Macro ***/
static constexpr float kResultMixRatio = 0.5f;
static constexpr bool  kIsDrawAllResult = true;

#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<SegmentationEngine> s_engine;
CommonHelper::NiceColorGenerator s_nice_color_generator(16);

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


int32_t ImageProcessor::Initialize(const InputParam& input_param)
{
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new SegmentationEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != SegmentationEngine::kRetOk) {
        s_engine->Finalize();
        s_engine.reset();
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_engine->Finalize() != SegmentationEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_engine) {
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


int32_t ImageProcessor::Process(cv::Mat& mat, Result& result)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    cv::resize(mat, mat, cv::Size(640, 640 * mat.rows / mat.cols));

    SegmentationEngine::Result segmentation_result;
    if (s_engine->Process(mat, segmentation_result) != SegmentationEngine::kRetOk) {
        return -1;
    }

    /* Draw segmentation image for all the classes weighted by score */
    cv::Mat mat_all_class = cv::Mat::zeros(segmentation_result.mat_out_list[0].size(), CV_8UC3);
    if (kIsDrawAllResult) {
        /* Pile all class */
#pragma omp parallel for
        for (int32_t i = 0; i < segmentation_result.mat_out_list.size(); i++) {
            auto& mat_out = segmentation_result.mat_out_list[i];
            cv::cvtColor(mat_out, mat_out, cv::COLOR_GRAY2BGR); /* 1channel -> 3 channel */
            cv::multiply(mat_out, s_nice_color_generator.Get(i), mat_out);
            mat_out.convertTo(mat_out, CV_8UC1);
        }

        // don't use parallel
        for (int32_t i = 0; i < segmentation_result.mat_out_list.size(); i++) {
            cv::add(mat_all_class, segmentation_result.mat_out_list[i], mat_all_class);
        }
    }

    /* Draw segmentation image for the class of the highest score */
    cv::Mat& mat_max = segmentation_result.mat_out_max;
    mat_max *= 255 / 19;    // to get nice color
    cv::applyColorMap(mat_max, mat_max, cv::COLORMAP_JET);

    /* Create result image */
    cv::resize(mat_max, mat_max, mat.size());
    cv::Mat mat_masked;
    cv::add(mat_max * kResultMixRatio, mat * (1.0f - kResultMixRatio), mat_masked);
    cv::hconcat(mat, mat_masked, mat);
    if (kIsDrawAllResult) {
        cv::resize(mat_all_class, mat_all_class, mat_max.size());
        cv::hconcat(mat_all_class, mat_max, mat_all_class);
        cv::vconcat(mat, mat_all_class, mat);
    }
    DrawFps(mat, segmentation_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = segmentation_result.time_pre_process;
    result.time_inference = segmentation_result.time_inference;
    result.time_post_process = segmentation_result.time_post_process;

    return 0;
}

