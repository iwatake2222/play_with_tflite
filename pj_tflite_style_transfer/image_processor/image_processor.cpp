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
#include "style_prediction_engine.h"
#include "style_transfer_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<StylePredictionEngine> s_style_prediction_engine;
std::unique_ptr<StyleTransferEngine> s_style_transfer_engine;
float s_style_bottleneck[StylePredictionEngine::SIZE_STYLE_BOTTLENECK];
std::string s_work_dir;
bool s_style_bottleneck_updated = true;

/*** Function ***/
static cv::Scalar CreateCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

static int32_t CalculateStyleBottleneck(std::string style_filename)
{
	std::string path = s_work_dir + "/style/" + style_filename;
	cv::Mat style_image = cv::imread(path);
	if (style_image.empty()) {
		PRINT("[error] cannot read %s\n", path.c_str());
		return -1;
	}

	StylePredictionEngine::Result style_prediction_result;
	s_style_prediction_engine->Process(style_image, style_prediction_result);
	for (int32_t i = 0; i < StylePredictionEngine::SIZE_STYLE_BOTTLENECK; i++) {
		s_style_bottleneck[i] = style_prediction_result.styleBottleneck[i];
	}
	s_style_bottleneck_updated = true;
	return 0;
}

int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam* input_param)
{
	if (s_style_prediction_engine || s_style_transfer_engine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_work_dir = input_param->work_dir;

	s_style_prediction_engine.reset(new StylePredictionEngine());
	if (s_style_prediction_engine->Initialize(input_param->work_dir, input_param->num_threads) != StylePredictionEngine::kRetOk) {
		s_style_prediction_engine->Finalize();
		s_style_prediction_engine.reset();
		return -1;
	}

	s_style_transfer_engine.reset(new StyleTransferEngine());
	if (s_style_transfer_engine->Initialize(input_param->work_dir, input_param->num_threads) != StyleTransferEngine::kRetOk) {
		s_style_transfer_engine->Finalize();
		s_style_transfer_engine.reset();
		return -1;
	}

	ImageProcessor::Command(0);

	return 0;
}

int32_t ImageProcessor::Finalize(void)
{
	if (!s_style_prediction_engine || !s_style_transfer_engine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_style_prediction_engine->Finalize() != StylePredictionEngine::kRetOk) {
		return -1;
	}

	if (s_style_transfer_engine->Finalize() != StyleTransferEngine::kRetOk) {
		return -1;
	}

	return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
	if (!s_style_prediction_engine || !s_style_transfer_engine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	static int32_t s_current_image_file_index = 0;
	switch (cmd) {
	case 0:
		s_current_image_file_index++;
		if (s_current_image_file_index > 30) s_current_image_file_index = 30;
		break;
	case 1:
		s_current_image_file_index--;
		if (s_current_image_file_index < 0) s_current_image_file_index = 0;
		break;
	case 2:
		s_current_image_file_index = 0;
		break;
	default:
		PRINT_E("command(%d) is not supported\n", cmd);
		return -1;
	}
	std::string filename = "style" + std::to_string(s_current_image_file_index) + ".jpg";
	CalculateStyleBottleneck(filename);

	return 0;
}


int32_t ImageProcessor::Process(cv::Mat* mat, ImageProcessor::OutputParam* output_param)
{
	if (!s_style_prediction_engine || !s_style_transfer_engine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	cv::Mat& original_mat = *mat;


	constexpr int32_t INTERVAL_TO_CALCULATE_CONTENT_BOTTLENECK = 10; // to increase FPS (no need to do this every frame)
	static float s_merged_style_bottleneck[StylePredictionEngine::SIZE_STYLE_BOTTLENECK];
	static int32_t s_cnt = 0;
	if (s_cnt++ % INTERVAL_TO_CALCULATE_CONTENT_BOTTLENECK == 0 || s_style_bottleneck_updated) {
		constexpr float ratio = 0.5f;
		StylePredictionEngine::Result style_prediction_result;
		s_style_prediction_engine->Process(original_mat, style_prediction_result);
		for (int32_t i = 0; i < StylePredictionEngine::SIZE_STYLE_BOTTLENECK; i++) {
			s_merged_style_bottleneck[i] = ratio * style_prediction_result.styleBottleneck[i] + (1 - ratio) * s_style_bottleneck[i];
		}
	}

	StyleTransferEngine::Result style_transfer_result;
	s_style_transfer_engine->Process(original_mat, s_merged_style_bottleneck, StylePredictionEngine::SIZE_STYLE_BOTTLENECK, style_transfer_result);

	/* Return the results */
	original_mat = style_transfer_result.image;
	output_param->time_pre_process = style_transfer_result.time_pre_process;
	output_param->time_inference = style_transfer_result.time_inference;
	output_param->time_post_process = style_transfer_result.time_post_process;

	return 0;
}

