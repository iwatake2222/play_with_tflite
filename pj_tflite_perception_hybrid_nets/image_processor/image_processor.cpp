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
#include "common_helper_cv.h"
#include "camera_model.h"
#include "bounding_box.h"
#include "detection_engine.h"
#include "tracker.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<DetectionEngine> s_engine;
Tracker s_tracker;
CommonHelper::NiceColorGenerator s_nice_color_generator;

/* For top view transform */
static CameraModel s_camera_real;
static CameraModel s_camera_top;
static cv::Mat s_mat_transform_topview;
#define COLOR_BG  CommonHelper::CreateCvColor(70, 70, 70)
static cv::Size s_size_topview;

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
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new DetectionEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != DetectionEngine::kRetOk) {
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

    if (s_engine->Finalize() != DetectionEngine::kRetOk) {
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

static void CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview)
{
    /* Perspective Transform */
    mat_topview = cv::Mat(cv::Size(s_camera_top.width, s_camera_top.height), CV_8UC3, COLOR_BG);
    //cv::warpPerspective(mat_original, mat_topview, s_mat_transform_topview, mat_topview.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    cv::warpPerspective(mat_original, mat_topview, s_mat_transform_topview, mat_topview.size(), cv::INTER_NEAREST);

#if 1
    /* Display Grid lines */
    static constexpr int32_t kDepthInterval = 5;
    static constexpr int32_t kHorizontalRange = 10;
    std::vector<cv::Point3f> object_point_list;
    for (float z = 0; z <= 30; z += kDepthInterval) {
        object_point_list.push_back(cv::Point3f(-kHorizontalRange, 0, z));
        object_point_list.push_back(cv::Point3f(kHorizontalRange, 0, z));
    }
    std::vector<cv::Point2f> image_point_list;
    cv::projectPoints(object_point_list, s_camera_top.rvec, s_camera_top.tvec, s_camera_top.K, s_camera_top.dist_coeff, image_point_list);
    for (int32_t i = 0; i < static_cast<int32_t>(image_point_list.size()); i++) {
        if (i % 2 != 0) {
            cv::line(mat_topview, image_point_list[i - 1], image_point_list[i], cv::Scalar(255, 255, 255));
        } else {
            CommonHelper::DrawText(mat_topview, std::to_string(i / 2 * kDepthInterval) + "[m]", image_point_list[i], 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);
        }
    }
#endif
}

static void CreateTransformMat(int32_t width, int32_t height, float fov_deg)
{
    /*** Set camera parameters ***/
    s_size_topview.width = width / 4;
    s_size_topview.height = height;
    s_camera_real.SetIntrinsic(width, height, FocalLength(width, fov_deg));
    s_camera_top.SetIntrinsic(s_size_topview.width, s_size_topview.height, FocalLength(s_size_topview.width, fov_deg));
    s_camera_real.SetExtrinsic(
        { 0.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, -1.5f, 0.0f }, true);   /* tvec (Oc - Ow in world coordinate. X+= Right, Y+ = down, Z+ = far) */
    s_camera_top.SetExtrinsic(
        { 90.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, -8.0f, 17.0f }, true);   /* tvec (Oc - Ow in world coordinate. X+= Right, Y+ = down, Z+ = far) */

    /*** Generate mapping b/w object points (3D: world coordinate) and image points (real camera) */
    std::vector<cv::Point3f> object_point_list = {   /* Target area (possible road area) */
        { -1.0f, 0, 10.0f },
        {  1.0f, 0, 10.0f },
        { -1.0f, 0,  3.0f },
        {  1.0f, 0,  3.0f },
    };
    std::vector<cv::Point2f> image_point_real_list;
    cv::projectPoints(object_point_list, s_camera_real.rvec, s_camera_real.tvec, s_camera_real.K, s_camera_real.dist_coeff, image_point_real_list);

    /* Convert to image points (2D) using the top view camera (virtual camera) */
    std::vector<cv::Point2f> image_point_top_list;
    cv::projectPoints(object_point_list, s_camera_top.rvec, s_camera_top.tvec, s_camera_top.K, s_camera_top.dist_coeff, image_point_top_list);

    s_mat_transform_topview = cv::getPerspectiveTransform(&image_point_real_list[0], &image_point_top_list[0]);
}


int32_t ImageProcessor::Process(cv::Mat& mat, ImageProcessor::Result& result)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    /*** Initialize camera parameters for input image size ***/
    static bool s_is_initialize_transform_mat = false;
    if (!s_is_initialize_transform_mat) {
        s_is_initialize_transform_mat = true;
        CreateTransformMat(mat.cols, mat.rows, 80);
    }

    /*** Call inference ***/
    DetectionEngine::Result det_result;
    if (s_engine->Process(mat, det_result) != DetectionEngine::kRetOk) {
        return -1;
    }

    /*** Draw target area  ***/
    cv::rectangle(mat, cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h), CommonHelper::CreateCvColor(0, 0, 0), 2);

    /*** Draw segmentation image for the class of the highest score ***/
    cv::Mat& mat_seg_max = det_result.mat_seg_max;
    cv::Mat mat_seg_max_list[] = { mat_seg_max, mat_seg_max, mat_seg_max };
    cv::merge(mat_seg_max_list, 3, mat_seg_max);
    cv::Mat mat_lut = cv::Mat::zeros(256, 1, CV_8UC3);
    mat_lut.at<cv::Vec3b>(0) = cv::Vec3b(0, 0, 0);
    mat_lut.at<cv::Vec3b>(1) = cv::Vec3b(0, 255, 0);
    mat_lut.at<cv::Vec3b>(2) = cv::Vec3b(0, 0, 255);
    cv::LUT(mat_seg_max, mat_lut, mat_seg_max);
    cv::resize(mat_seg_max, mat_seg_max, mat.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat mat_masked;
    cv::addWeighted(mat, 0.8, mat_seg_max, 0.5, 0, mat_masked);
    //cv::add(mat_seg_max * kResultMixRatio, mat * (1.0f - kResultMixRatio), mat_masked);
    //cv::hconcat(mat, mat_masked, mat);
    mat = mat_masked;

    /*** Draw detection result (black rectangle) ***/
    int32_t num_det = 0;
    for (const auto& bbox : det_result.bbox_list) {
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 0, 0), 1);
        num_det++;
    }

    /*** Draw tracking result ***/
    s_tracker.Update(det_result.bbox_list);
    int32_t num_track = 0;
    auto& track_list = s_tracker.GetTrackList();
    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : s_nice_color_generator.Get(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        CommonHelper::DrawText(mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y - 13), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        auto& track_history = track.GetDataHistory();
        for (size_t i = 1; i < track_history.size(); i++) {
            cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
            cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
            cv::line(mat, p0, p1, CommonHelper::CreateCvColor(255, 0, 0));
        }
        num_track++;
    }
    CommonHelper::DrawText(mat, "DET: " + std::to_string(num_det) + ", TRACK: " + std::to_string(num_track), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

    /*** Draw top view ***/
    cv::Mat mat_topview;
    CreateTopViewMat(mat_seg_max, mat_topview);
    /* Draw object on top view */
    std::vector<cv::Point2f> normal_points;
    std::vector<cv::Point2f> topview_points;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        normal_points.push_back({ bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f });
    }
    if (normal_points.size() > 0) {
        cv::perspectiveTransform(normal_points, topview_points, s_mat_transform_topview);
        for (int32_t i = 0; i < static_cast<int32_t>(track_list.size()); i++) {
            auto& track = track_list[i];
            const auto& bbox = track.GetLatestData().bbox;
            cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : s_nice_color_generator.Get(track.GetId());
            cv::Point p(static_cast<int32_t>(topview_points[i].x), static_cast<int32_t>(topview_points[i].y));
            cv::circle(mat_topview, p, 10, color, -1);
            cv::circle(mat_topview, p, 10, cv::Scalar(0, 0, 0), 2);
        }
    }
    cv::hconcat(mat, mat_topview, mat);

    DrawFps(mat, det_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    int32_t bbox_num = 0;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        result.object_list[bbox_num].class_id = bbox.class_id;
        snprintf(result.object_list[bbox_num].label, sizeof(result.object_list[bbox_num].label), "%s", bbox.label.c_str());
        result.object_list[bbox_num].score = bbox.score;
        result.object_list[bbox_num].x = bbox.x;
        result.object_list[bbox_num].y = bbox.y;
        result.object_list[bbox_num].width = bbox.w;
        result.object_list[bbox_num].height = bbox.h;
        bbox_num++;
        if (bbox_num >= NUM_MAX_RESULT) break;
    }
    result.object_num = bbox_num;

    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;

    return 0;
}

