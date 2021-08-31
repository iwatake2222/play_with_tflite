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
#include <numeric>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "dbscan.hpp"
#include "kdtree.h"

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "lane_engine.h"

/*** Macro ***/
#define TAG "LaneEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define USE_TFLITE
#ifdef USE_TFLITE
#define MODEL_NAME  "lanenet-lane-detection.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_tensor"
#define INPUT_DIMS  { 1, 256, 512, 3}
#define IS_NCHW     false
#define IS_RGB      false
#define OUTPUT_NAME_0 "LaneNet/bisenetv2_backend/binary_seg/ArgMax"
#define OUTPUT_NAME_1 "LaneNet/bisenetv2_backend/instance_seg/pix_embedding_conv/pix_embedding_conv"
#else
#define MODEL_NAME  "lanenet-lane-detection.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_tensor"
#define INPUT_DIMS  { 1, 256, 512, 3}
#define IS_NCHW     false
#define IS_RGB      false
#define OUTPUT_NAME_0 "binary_seg_ret"
#define OUTPUT_NAME_1 "instance_seg_ret"
#endif

static constexpr int32_t kNumWidth = 512;
static constexpr int32_t kNumHeight = 256;

/*** Function ***/
static void gather_pixel_embedding_features(const cv::Mat& binary_mask, const cv::Mat& pixel_embedding, std::vector<cv::Point>& coords, std::vector<DBSCAMSample<float>>& embedding_samples);
template <typename T1, typename T2>
static void simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2);
static Feature<float> calculate_stddev_feature_vector(const std::vector<DBSCAMSample<float>>& input_samples, const Feature<float>& mean_feature_vec);
static Feature<float> calculate_mean_feature_vector(const std::vector<DBSCAMSample<float>>& input_samples);
static void normalize_sample_features(const std::vector<DBSCAMSample<float>>& input_samples, std::vector<DBSCAMSample<float>>& output_samples);
static void cluster_pixem_embedding_features(std::vector<DBSCAMSample<float>>& embedding_samples, std::vector<std::vector<uint> >& cluster_ret, std::vector<uint>& noise);
static void visualize_instance_segmentation_result(const std::vector<std::vector<uint> >& cluster_ret, const std::vector<cv::Point>& coords, cv::Mat& intance_segmentation_result);


int32_t LaneEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* -1.0 - 1.0 */
    input_tensor_info.normalize.mean[0] = 0.5f;
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
#ifdef USE_TFLITE
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TensorInfo::kTensorTypeInt64));
#else
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TensorInfo::kTensorTypeInt32));
#endif
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));

    /* Create and Initialize Inference Helper */
#ifdef USE_TFLITE
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#else
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));
#endif

    if (!inference_helper_) {
        return kRetErr;
    }
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

int32_t LaneEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t LaneEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    /* do resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = crop_w / 2;
    int32_t crop_y = original_mat.rows - crop_h ;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;

    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();

    /* Retrieve the result */
#ifdef USE_TFLITE
    std::vector<int64_t> output_binary(static_cast<int64_t*>(output_tensor_info_list_[0].data), static_cast<int64_t*>(output_tensor_info_list_[0].data) + output_tensor_info_list_[0].GetElementNum());
#else
    std::vector<int32_t> output_binary(static_cast<int32_t*>(output_tensor_info_list_[0].data), static_cast<int32_t*>(output_tensor_info_list_[0].data) + output_tensor_info_list_[0].GetElementNum());
#endif
    std::vector<float> output_pixel_embedding(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());

    cv::Mat image_binary(kNumHeight, kNumWidth, CV_8UC1);
    for (int32_t i = 0; i < output_binary.size(); i++) {
        image_binary.data[i] = output_binary[i] * 255;
    }

    cv::Mat image_instance(kNumHeight, kNumWidth, CV_32FC4, output_pixel_embedding.data());

    std::vector<cv::Point> coords;
    std::vector<DBSCAMSample<float>> pixel_embedding_samples;
    gather_pixel_embedding_features(image_binary, image_instance, coords, pixel_embedding_samples);

    simultaneously_random_shuffle<cv::Point, DBSCAMSample<float>>(coords, pixel_embedding_samples);
    normalize_sample_features(pixel_embedding_samples, pixel_embedding_samples);
    std::vector<std::vector<uint> > cluster_ret;
    std::vector<uint> noise;
    cluster_pixem_embedding_features(pixel_embedding_samples, cluster_ret, noise);
    cv::Mat instance_seg_result = cv::Mat(kNumHeight, kNumWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    visualize_instance_segmentation_result(cluster_ret, coords, instance_seg_result);


    cv::resize(image_binary, image_binary, cv::Size(crop_w, crop_h));
    result.image_binary_seg = cv::Mat(original_mat.size(), CV_8UC1);
    cv::Rect crop = cv::Rect(crop_x < 0 ? 0 : crop_x, crop_y < 0 ? 0 : crop_y, crop_x < 0 ? original_mat.cols : crop_w, crop_h < 0 ? original_mat.rows : crop_h);
    cv::Mat target = result.image_binary_seg(crop);
    image_binary(cv::Rect(crop_x < 0 ? -crop_x : 0, crop_y < 0 ? -crop_y : 0, crop_x < 0 ? original_mat.cols : crop_w, crop_h < 0 ? original_mat.rows : crop_h)).copyTo(target);

    cv::resize(instance_seg_result, instance_seg_result, cv::Size(crop_w, crop_h));
    result.image_instance_seg = cv::Mat(original_mat.size(), CV_8UC3);
    crop = cv::Rect(crop_x < 0 ? 0 : crop_x, crop_y < 0 ? 0 : crop_y, crop_x < 0 ? original_mat.cols : crop_w, crop_h < 0 ? original_mat.rows : crop_h);
    target = result.image_instance_seg(crop);
    instance_seg_result(cv::Rect(crop_x < 0 ? -crop_x : 0, crop_y < 0 ? -crop_y : 0, crop_x < 0 ? original_mat.cols : crop_w, crop_h < 0 ? original_mat.rows : crop_h)).copyTo(target);


    const auto& t_post_process1 = std::chrono::steady_clock::now();


    /* Return the results */
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}


/***************************************************************************************************
 * The following code is retrieved from https://github.com/xuanyuyt/lanenet-lane-detection
 ***************************************************************************************************/
/************************************************
* Copyright 2019 Baidu Inc. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lanenetModel.cpp
* Date: 2019/11/5 ‰ºŒß5:19
************************************************/
static void gather_pixel_embedding_features(const cv::Mat& binary_mask, const cv::Mat& pixel_embedding,
    std::vector<cv::Point>& coords,
    std::vector<DBSCAMSample<float>>& embedding_samples) {


    auto image_rows = kNumHeight;
    auto image_cols = kNumWidth;

    for (auto row = 0; row < image_rows; ++row) {
        auto binary_image_row_data = binary_mask.ptr<uchar>(row);
        auto embedding_image_row_data = pixel_embedding.ptr<cv::Vec4f>(row);
        for (auto col = 0; col < image_cols; ++col) {
            auto binary_image_pix_value = binary_image_row_data[col];
            if (binary_image_pix_value == 255) {
                coords.emplace_back(cv::Point(col, row));
                Feature<float> embedding_features;
                for (auto index = 0; index < 4; ++index) {
                    embedding_features.push_back(embedding_image_row_data[col][index]);
                }
                DBSCAMSample<float> sample(embedding_features, CLASSIFY_FLAGS::NOT_CALSSIFIED);
                embedding_samples.push_back(sample);
            }
        }
    }
}

template <typename T1, typename T2>
static void simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2) {

    if (src1.empty() || src2.empty()) {
        return;
    }

    // construct index vector of two input src
    std::vector<uint> indexes;
    indexes.reserve(src1.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::random_shuffle(indexes.begin(), indexes.end());

    // make copy of two input vector
    std::vector<T1> src1_copy(src1);
    std::vector<T2> src2_copy(src2);

    // random two source input vector via random shuffled index vector
    for (uint i = 0; i < indexes.size(); ++i) {
        src1[i] = src1_copy[indexes[i]];
        src2[i] = src2_copy[indexes[i]];
    }
}

static Feature<float> calculate_stddev_feature_vector(
    const std::vector<DBSCAMSample<float>>& input_samples,
    const Feature<float>& mean_feature_vec) {

    if (input_samples.empty()) {
        return Feature<float>();
    }

    auto feature_dims = input_samples[0].get_feature_vector().size();
    auto sample_nums = input_samples.size();

    // calculate stddev feature vector
    Feature<float> stddev_feature_vec;
    stddev_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (auto index = 0; index < feature_dims; ++index) {
            auto sample_feature = sample.get_feature_vector();
            auto diff = sample_feature[index] - mean_feature_vec[index];
            diff = std::pow(diff, 2);
            stddev_feature_vec[index] += diff;
        }
    }
    for (auto index = 0; index < feature_dims; ++index) {
        stddev_feature_vec[index] /= sample_nums;
        stddev_feature_vec[index] = std::sqrt(stddev_feature_vec[index]);
    }

    return stddev_feature_vec;
}

static Feature<float> calculate_mean_feature_vector(const std::vector<DBSCAMSample<float>>& input_samples) {

    if (input_samples.empty()) {
        return Feature<float>();
    }

    auto feature_dims = input_samples[0].get_feature_vector().size();
    auto sample_nums = input_samples.size();
    Feature<float> mean_feature_vec;
    mean_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (auto index = 0; index < feature_dims; ++index) {
            mean_feature_vec[index] += sample[index];
        }
    }
    for (auto index = 0; index < feature_dims; ++index) {
        mean_feature_vec[index] /= sample_nums;
    }

    return mean_feature_vec;
}

static void normalize_sample_features(const std::vector<DBSCAMSample<float>>& input_samples,
    std::vector<DBSCAMSample<float>>& output_samples) {
    // calcualte mean feature vector
    Feature<float> mean_feature_vector = calculate_mean_feature_vector(input_samples);

    // calculate stddev feature vector
    Feature<float> stddev_feature_vector = calculate_stddev_feature_vector(input_samples, mean_feature_vector);

    std::vector<DBSCAMSample<float>> input_samples_copy = input_samples;
    for (auto& sample : input_samples_copy) {
        auto feature = sample.get_feature_vector();
        for (auto index = 0; index < feature.size(); ++index) {
            feature[index] = (feature[index] - mean_feature_vector[index]) / stddev_feature_vector[index];
        }
        sample.set_feature_vector(feature);
    }
    output_samples = input_samples_copy;
}

static void cluster_pixem_embedding_features(std::vector<DBSCAMSample<float>>& embedding_samples,
    std::vector<std::vector<uint> >& cluster_ret, std::vector<uint>& noise) {

    if (embedding_samples.empty()) {
        PRINT_E("Pixel embedding samples empty")
            return;
    }

    // dbscan cluster
    auto dbscan = DBSCAN<DBSCAMSample<float>, float>();
    dbscan.Run(&embedding_samples, 4, 0.4, 500);            /* from config.ini */
    cluster_ret = dbscan.Clusters;
    noise = dbscan.Noise;
}

static void visualize_instance_segmentation_result(
    const std::vector<std::vector<uint> >& cluster_ret,
    const std::vector<cv::Point>& coords,
    cv::Mat& intance_segmentation_result) {

    std::map<int, cv::Scalar> color_map = {
        {0, cv::Scalar(0, 0, 255)},
        {1, cv::Scalar(0, 255, 0)},
        {2, cv::Scalar(255, 0, 0)},
        {3, cv::Scalar(255, 0, 255)},
        {4, cv::Scalar(0, 255, 255)},
        {5, cv::Scalar(255, 255, 0)},
        {6, cv::Scalar(125, 0, 125)},
        {7, cv::Scalar(0, 125, 125)}
    };

    for (int class_id = 0; class_id < cluster_ret.size(); ++class_id) {
        auto class_color = color_map[class_id];
#pragma omp parallel for
        for (auto index = 0; index < cluster_ret[class_id].size(); ++index) {
            auto coord = coords[cluster_ret[class_id][index]];
            auto image_col_data = intance_segmentation_result.ptr<cv::Vec3b>(coord.y);
            image_col_data[coord.x][0] = class_color[0];
            image_col_data[coord.x][1] = class_color[1];
            image_col_data[coord.x][2] = class_color[2];
        }
    }
}
