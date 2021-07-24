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
#include <fstream>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "pose_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<PoseEngine> s_pose_engine;

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
    if (s_pose_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_pose_engine.reset(new PoseEngine());
    if (s_pose_engine->Initialize(input_param->work_dir, input_param->num_threads) != PoseEngine::kRetOk) {
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_pose_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_pose_engine->Finalize() != PoseEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_pose_engine) {
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

static const std::vector<std::pair<int32_t, int32_t>> jointLineList {
    /* face */
    {0, 2},
    {2, 4},
    {0, 1},
    {1, 3},
    /* body */
    {6, 5},
    {5, 11},
    {11, 12},
    {12, 6},
    /* arm */
    {6, 8},
    {8, 10},
    {5, 7},
    {7, 9},
    /* leg */
    {12, 14},
    {14, 16},
    {11, 13},
    {13, 15},
};

int32_t ImageProcessor::Process(cv::Mat* mat, ImageProcessor::OutputParam* output_param)
{
    if (!s_pose_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    cv::Mat& original_mat = *mat;
    PoseEngine::Result result;
    if (s_pose_engine->Process(original_mat, result) != PoseEngine::kRetOk) {
        return -1;
    }

    /* Draw the result */
    /* note: we have only one body with this model */
    constexpr float score_threshold = 0.2F;
    const auto& score_list = result.pose_keypoint_scores[0];
    const auto& part_list = result.pose_keypoint_coords[0];
    int32_t part_num = part_list.size();

    for (const auto& jointLine : jointLineList) {
        if (score_list[jointLine.first] >= score_threshold && score_list[jointLine.second] >= score_threshold) {
            int32_t x0 = static_cast<int32_t>(part_list[jointLine.first].first * original_mat.cols);
            int32_t y0 = static_cast<int32_t>(part_list[jointLine.first].second * original_mat.rows);
            int32_t x1 = static_cast<int32_t>(part_list[jointLine.second].first * original_mat.cols);
            int32_t y1 = static_cast<int32_t>(part_list[jointLine.second].second * original_mat.rows);
            cv::line(original_mat, cv::Point(x0, y0), cv::Point(x1, y1) , CreateCvColor(200, 200, 200), 2);
        }
    }

    for (int32_t i = 0; i < part_num; i++) {
        int32_t x = static_cast<int32_t>(part_list[i].first * original_mat.cols);
        int32_t y = static_cast<int32_t>(part_list[i].second * original_mat.rows);
        float score = score_list[i];
        if (score >= score_threshold) {
            cv::circle(original_mat, cv::Point(x, y), 5, CreateCvColor(0, 255, 0), -1);
        }
    }

    /* Return the results */
    output_param->time_pre_process = result.time_pre_process;
    output_param->time_inference = result.time_inference;
    output_param->time_post_process = result.time_post_process;

    return 0;
}

