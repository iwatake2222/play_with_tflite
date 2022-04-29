/* Copyright 2022 iwatake2222

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
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
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

#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<SegmentationEngine> s_engine;
cv::Mat s_mat_lut;
extern std::vector<std::array<uint8_t, 3>> s_palette;

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

    /* Create LUT */
    std::vector<uint8_t> seq_num(256);
    std::iota(seq_num.begin(), seq_num.end(), 0);
    std::mt19937 get_rand_mt(0);
    std::shuffle(seq_num.begin(), seq_num.end(), get_rand_mt);
    cv::Mat mat_seq = cv::Mat(256, 1, CV_8UC1, seq_num.data());
    cv::Mat mat_colormap;
    cv::applyColorMap(mat_seq, mat_colormap, cv::COLORMAP_RAINBOW);
    s_mat_lut = cv::Mat::zeros(256, 1, CV_8UC3);
    for (int32_t i = 0; i < 256; i++) {
        s_mat_lut.at<cv::Vec3b>(i) = mat_colormap.at<cv::Vec3b>(i);
    }

#if 1
    for (size_t i = 0; i < s_palette.size(); i++) {
        s_mat_lut.at<cv::Vec3b>(i)[0] = s_palette[i][0];
        s_mat_lut.at<cv::Vec3b>(i)[1] = s_palette[i][1];
        s_mat_lut.at<cv::Vec3b>(i)[2] = s_palette[i][2];
    }
#endif

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

    /* Draw segmentation image for the class of the highest score */
    cv::Mat& mat_seg_max = segmentation_result.mat_out_max;

#if 1
    cv::Mat mat_seg_max_list[] = { mat_seg_max, mat_seg_max, mat_seg_max };
    cv::merge(mat_seg_max_list, 3, mat_seg_max);
    cv::LUT(mat_seg_max, s_mat_lut, mat_seg_max);
#else
    mat_seg_max *= 255 / 150;    // to get nice color
    cv::applyColorMap(mat_seg_max, mat_seg_max, cv::COLORMAP_JET);
#endif

    /* Create result image */
    cv::resize(mat_seg_max, mat_seg_max, mat.size());
    cv::Mat mat_masked;
    cv::add(mat_seg_max * kResultMixRatio, mat * (1.0f - kResultMixRatio), mat_masked);
    
    DrawFps(mat_masked, segmentation_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    mat = mat_masked;

    /* Return the results */
    result.time_pre_process = segmentation_result.time_pre_process;
    result.time_inference = segmentation_result.time_inference;
    result.time_post_process = segmentation_result.time_post_process;

    return 0;
}

std::vector<std::array<uint8_t,3>> s_palette = 
{
    {120, 120, 120}, { 180, 120, 120 }, { 6, 230, 230 }, { 80, 50, 50 },
    { 4, 200, 3 }, { 120, 120, 80 }, { 140, 140, 140 }, { 204, 5, 255 },
    { 230, 230, 230 }, { 4, 250, 7 }, { 224, 5, 255 }, { 235, 255, 7 },
    { 150, 5, 61 }, { 120, 120, 70 }, { 8, 255, 51 }, { 255, 6, 82 },
    { 143, 255, 140 }, { 204, 255, 4 }, { 255, 51, 7 }, { 204, 70, 3 },
    { 0, 102, 200 }, { 61, 230, 250 }, { 255, 6, 51 }, { 11, 102, 255 },
    { 255, 7, 71 }, { 255, 9, 224 }, { 9, 7, 230 }, { 220, 220, 220 },
    { 255, 9, 92 }, { 112, 9, 255 }, { 8, 255, 214 }, { 7, 255, 224 },
    { 255, 184, 6 }, { 10, 255, 71 }, { 255, 41, 10 }, { 7, 255, 255 },
    { 224, 255, 8 }, { 102, 8, 255 }, { 255, 61, 6 }, { 255, 194, 7 },
    { 255, 122, 8 }, { 0, 255, 20 }, { 255, 8, 41 }, { 255, 5, 153 },
    { 6, 51, 255 }, { 235, 12, 255 }, { 160, 150, 20 }, { 0, 163, 255 },
    { 140, 140, 140 }, { 250, 10, 15 }, { 20, 255, 0 }, { 31, 255, 0 },
    { 255, 31, 0 }, { 255, 224, 0 }, { 153, 255, 0 }, { 0, 0, 255 },
    { 255, 71, 0 }, { 0, 235, 255 }, { 0, 173, 255 }, { 31, 0, 255 },
    { 11, 200, 200 }, { 255, 82, 0 }, { 0, 255, 245 }, { 0, 61, 255 },
    { 0, 255, 112 }, { 0, 255, 133 }, { 255, 0, 0 }, { 255, 163, 0 },
    { 255, 102, 0 }, { 194, 255, 0 }, { 0, 143, 255 }, { 51, 255, 0 },
    { 0, 82, 255 }, { 0, 255, 41 }, { 0, 255, 173 }, { 10, 0, 255 },
    { 173, 255, 0 }, { 0, 255, 153 }, { 255, 92, 0 }, { 255, 0, 255 },
    { 255, 0, 245 }, { 255, 0, 102 }, { 255, 173, 0 }, { 255, 0, 20 },
    { 255, 184, 184 }, { 0, 31, 255 }, { 0, 255, 61 }, { 0, 71, 255 },
    { 255, 0, 204 }, { 0, 255, 194 }, { 0, 255, 82 }, { 0, 10, 255 },
    { 0, 112, 255 }, { 51, 0, 255 }, { 0, 194, 255 }, { 0, 122, 255 },
    { 0, 255, 163 }, { 255, 153, 0 }, { 0, 255, 10 }, { 255, 112, 0 },
    { 143, 255, 0 }, { 82, 0, 255 }, { 163, 255, 0 }, { 255, 235, 0 },
    { 8, 184, 170 }, { 133, 0, 255 }, { 0, 255, 92 }, { 184, 0, 255 },
    { 255, 0, 31 }, { 0, 184, 255 }, { 0, 214, 255 }, { 255, 0, 112 },
    { 92, 255, 0 }, { 0, 224, 255 }, { 112, 224, 255 }, { 70, 184, 160 },
    { 163, 0, 255 }, { 153, 0, 255 }, { 71, 255, 0 }, { 255, 0, 163 },
    { 255, 204, 0 }, { 255, 0, 143 }, { 0, 255, 235 }, { 133, 255, 0 },
    { 255, 0, 235 }, { 245, 0, 255 }, { 255, 0, 122 }, { 255, 245, 0 },
    { 10, 190, 212 }, { 214, 255, 0 }, { 0, 204, 255 }, { 20, 0, 255 },
    { 255, 255, 0 }, { 0, 153, 255 }, { 0, 41, 255 }, { 0, 255, 204 },
    { 41, 0, 255 }, { 41, 255, 0 }, { 173, 0, 255 }, { 0, 245, 255 },
    { 71, 0, 255 }, { 122, 0, 255 }, { 0, 255, 184 }, { 0, 92, 255 },
    { 184, 255, 0 }, { 0, 133, 255 }, { 255, 214, 0 }, { 25, 194, 194 },
    { 102, 255, 0 }, { 92, 0, 255 }};