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
#include "bounding_box.h"
#include "face_detection_engine.h"
#include "headpose_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<FaceDetectionEngine> s_facedet_engine;
std::unique_ptr<HeadposeEngine> s_headpose_engine;
cv::Mat s_camera_matrix;

/*** Function ***/
static cv::Scalar CreateCvColor(int32_t b, int32_t g, int32_t r)
{
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}

static void DrawText(cv::Mat& mat, const std::string& text, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    int32_t baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    baseline += thickness;
    pos.y += textSize.height;
    if (is_text_on_rect) {
        cv::rectangle(mat, pos + cv::Point(0, baseline), pos + cv::Point(textSize.width, -textSize.height), color_back, -1);
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_front, thickness);
    } else {
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_back, thickness * 3);
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_front, thickness);
    }
}

static void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
    DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CreateCvColor(0, 0, 0), CreateCvColor(180, 180, 180), true);
}

int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam& input_param)
{
    if (s_facedet_engine || s_headpose_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_facedet_engine.reset(new FaceDetectionEngine());
    if (s_facedet_engine->Initialize(input_param.work_dir, input_param.num_threads) != FaceDetectionEngine::kRetOk) {
        s_facedet_engine->Finalize();
        s_facedet_engine.reset();
        return -1;
    }

    s_headpose_engine.reset(new HeadposeEngine());
    if (s_headpose_engine->Initialize(input_param.work_dir, input_param.num_threads) != HeadposeEngine::kRetOk) {
        s_headpose_engine->Finalize();
        s_headpose_engine.reset();
        return -1;
    }

    s_camera_matrix.release();

    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_facedet_engine || !s_headpose_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_facedet_engine->Finalize() != HeadposeEngine::kRetOk) {
        return -1;
    }

    if (s_headpose_engine->Finalize() != HeadposeEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_facedet_engine || !s_headpose_engine) {
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

/* reference: https://github.com/incluit/OpenVino-Driver-Behaviour/blob/master/src/detectors.cpp#L474 */
static cv::Mat BuildCameraMatrix(int cx, int cy, float fx, float fy)
{
    /*
    fx 0 cx
    fy 0 cy
     0 0  1
    */
    cv::Mat s_camera_matrix = cv::Mat::zeros(3, 3, CV_32F);
    s_camera_matrix.at<float>(0) = fx;
    s_camera_matrix.at<float>(2) = static_cast<float>(cx);
    s_camera_matrix.at<float>(4) = fy;
    s_camera_matrix.at<float>(5) = static_cast<float>(cy);
    s_camera_matrix.at<float>(8) = 1;
    return s_camera_matrix;
}

/* reference:  https://github.com/incluit/OpenVino-Driver-Behaviour/blob/master/src/detectors.cpp#L484 */
static void DrawHeadPoseAxes(cv::Mat& mat, const cv::Mat& camera_matrix, const cv::Point& cpoint, float yaw, float pitch, float roll, float scale)
{
    pitch *= static_cast<float>(CV_PI / 180.0);
    yaw *= static_cast<float>(CV_PI / 180.0);
    roll *= static_cast<float>(CV_PI / 180.0);

    cv::Matx33f        Rx(1, 0, 0,
        0, cos(pitch), -sin(pitch),
        0, sin(pitch), cos(pitch));
    cv::Matx33f Ry(cos(yaw), 0, -sin(yaw),
        0, 1, 0,
        sin(yaw), 0, cos(yaw));
    cv::Matx33f Rz(cos(roll), -sin(roll), 0,
        sin(roll), cos(roll), 0,
        0, 0, 1);

    auto r = cv::Mat(Rz * Ry * Rx);

    cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

    xAxis.at<float>(0) = 1 * scale;
    xAxis.at<float>(1) = 0;
    xAxis.at<float>(2) = 0;

    yAxis.at<float>(0) = 0;
    yAxis.at<float>(1) = -1 * scale;
    yAxis.at<float>(2) = 0;

    zAxis.at<float>(0) = 0;
    zAxis.at<float>(1) = 0;
    zAxis.at<float>(2) = -1 * scale;

    zAxis1.at<float>(0) = 0;
    zAxis1.at<float>(1) = 0;
    zAxis1.at<float>(2) = 1 * scale;

    cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
    o.at<float>(2) = s_camera_matrix.at<float>(0);

    xAxis = r * xAxis + o;
    yAxis = r * yAxis + o;
    zAxis = r * zAxis + o;
    zAxis1 = r * zAxis1 + o;

    cv::Point p1, p2;

    p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * s_camera_matrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * s_camera_matrix.at<float>(4)) + cpoint.y);
    cv::line(mat, cv::Point(cpoint.x, cpoint.y), p2, CreateCvColor(0, 0, 255), 2);

    p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * s_camera_matrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * s_camera_matrix.at<float>(4)) + cpoint.y);
    cv::line(mat, cv::Point(cpoint.x, cpoint.y), p2, CreateCvColor(0, 255, 0), 2);

    p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * s_camera_matrix.at<float>(0)) + cpoint.x);
    p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * s_camera_matrix.at<float>(4)) + cpoint.y);

    p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * s_camera_matrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * s_camera_matrix.at<float>(4)) + cpoint.y);
    //cv::line(mat, p1, p2, CreateCvColor(255, 0, 0), 2);
    cv::line(mat, cv::Point(cpoint.x, cpoint.y), p2, CreateCvColor(255, 0, 0), 2);
}

static float CalcFocalLength(int32_t pixel_size, float fov)
{
    fov *= static_cast<float>(CV_PI / 180.0);
    return (pixel_size / 2) / tanf(fov / 2);
}

int32_t ImageProcessor::Process(cv::Mat& mat, ImageProcessor::Result& result)
{
    if (!s_facedet_engine || !s_headpose_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    /* Calculate camera matrix if not ready yet */
    if (s_camera_matrix.empty()) {
        s_camera_matrix = BuildCameraMatrix(mat.cols / 2, mat.rows / 2, CalcFocalLength(mat.cols, 80), CalcFocalLength(mat.rows, 80));
    }

    /* Detect face */
    FaceDetectionEngine::Result det_result;
    if (s_facedet_engine->Process(mat, det_result) != FaceDetectionEngine::kRetOk) {
        return -1;
    }

    /* Display target area  */
    cv::rectangle(mat, cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h), CreateCvColor(0, 0, 0), 2);

    /* Display detection result (black rectangle) */
    int32_t num_det = 0;
    for (const auto& bbox : det_result.bbox_list) {
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CreateCvColor(0, 0, 0), 1);
        num_det++;
    }

    for (const auto& keypoint : det_result.keypoint_list) {
        for (const auto& p : keypoint) {
            cv::circle(mat, cv::Point(p.first, p.second), 1, CreateCvColor(0, 255, 0));
        }
    }

    /* Use fixed value for test image */
    std::vector<BoundingBox> bbox_list = det_result.bbox_list;


    /* Estimate head pose */
    std::vector<HeadposeEngine::Result> headpose_result_list;
    if (s_headpose_engine->Process(mat, bbox_list, headpose_result_list) != HeadposeEngine::kRetOk) {
        return -1;
    }

    /* Display head poses */
    for (int32_t i = 0; i < (std::min)(bbox_list.size(), headpose_result_list.size()); i++) {
        cv::rectangle(mat, cv::Rect(bbox_list[i].x, bbox_list[i].y, bbox_list[i].w, bbox_list[i].h), CreateCvColor(0, 200, 0), 1);
        PRINT("%f %f %f\n", headpose_result_list[i].yaw, headpose_result_list[i].pitch, headpose_result_list[i].roll);
        cv::Point cpoint(bbox_list[i].x + bbox_list[i].w / 2, bbox_list[i].y + bbox_list[i].h / 2);
        float line_scale = bbox_list[i].h * 0.8f;
        DrawHeadPoseAxes(mat, s_camera_matrix, cpoint, headpose_result_list[i].yaw, headpose_result_list[i].pitch, headpose_result_list[i].roll, line_scale);
    }

    DrawText(mat, "DET: " + std::to_string(bbox_list.size()), cv::Point(0, 20), 0.7, 2, CreateCvColor(0, 0, 0), CreateCvColor(220, 220, 220));

    /* Return the results */
    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;
    for (const auto& headpose_result : headpose_result_list) {
        result.time_pre_process += headpose_result.time_pre_process;
        result.time_inference += headpose_result.time_inference;
        result.time_post_process += headpose_result.time_post_process;
    }

    DrawFps(mat, result.time_inference, cv::Point(0, 0), 0.5, 2, CreateCvColor(0, 0, 0), CreateCvColor(180, 180, 180), true);

    return 0;
}

