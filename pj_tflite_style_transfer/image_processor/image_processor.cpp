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
std::unique_ptr<StylePredictionEngine> s_stylePredictionEngine;
std::unique_ptr<StyleTransferEngine> s_styleTransferEngine;
float s_styleBottleneck[StylePredictionEngine::SIZE_STYLE_BOTTLENECK];
std::string s_workDir;
bool s_styleBottleneckUpdated = true;

/*** Function ***/
static cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

static int32_t calculateStyleBottleneck(std::string styleFilename)
{
	std::string path = s_workDir + "/style/" + styleFilename;
	cv::Mat styleImage = cv::imread(path);
	if (styleImage.empty()) {
		PRINT("[error] cannot read %s\n", path.c_str());
		return -1;
	}

	StylePredictionEngine::RESULT stylePredictionResult;
	s_stylePredictionEngine->invoke(styleImage, stylePredictionResult);
	for (int32_t i = 0; i < StylePredictionEngine::SIZE_STYLE_BOTTLENECK; i++) {
		s_styleBottleneck[i] = stylePredictionResult.styleBottleneck[i];
	}
	s_styleBottleneckUpdated = true;
	return 0;
}

int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam)
{
	if (s_stylePredictionEngine || s_styleTransferEngine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_workDir = inputParam->workDir;

	s_stylePredictionEngine.reset(new StylePredictionEngine());
	if (s_stylePredictionEngine->initialize(inputParam->workDir, inputParam->numThreads) != StylePredictionEngine::RET_OK) {
		s_stylePredictionEngine->finalize();
		s_stylePredictionEngine.reset();
		return -1;
	}

	s_styleTransferEngine.reset(new StyleTransferEngine());
	if (s_styleTransferEngine->initialize(inputParam->workDir, inputParam->numThreads) != StyleTransferEngine::RET_OK) {
		s_styleTransferEngine->finalize();
		s_styleTransferEngine.reset();
		return -1;
	}

	ImageProcessor_command(0);

	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_stylePredictionEngine || !s_styleTransferEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_stylePredictionEngine->finalize() != StylePredictionEngine::RET_OK) {
		return -1;
	}

	if (s_styleTransferEngine->finalize() != StyleTransferEngine::RET_OK) {
		return -1;
	}

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_stylePredictionEngine || !s_styleTransferEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	static int s_currentImageFileIndex = 0;
	switch (cmd) {
	case 0:
		s_currentImageFileIndex++;
		if (s_currentImageFileIndex > 30) s_currentImageFileIndex = 30;
		break;
	case 1:
		s_currentImageFileIndex--;
		if (s_currentImageFileIndex < 0) s_currentImageFileIndex = 0;
		break;
	case 2:
		s_currentImageFileIndex = 0;
		break;
	default:
		PRINT_E("command(%d) is not supported\n", cmd);
		return -1;
	}
	std::string filename = "style" + std::to_string(s_currentImageFileIndex) + ".jpg";
	calculateStyleBottleneck(filename);

	return 0;
}


int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam)
{
	if (!s_stylePredictionEngine || !s_styleTransferEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	cv::Mat& originalMat = *mat;


	constexpr int32_t INTERVAL_TO_CALCULATE_CONTENT_BOTTLENECK = 10; // to increase FPS (no need to do this every frame)
	static float s_mergedStyleBottleneck[StylePredictionEngine::SIZE_STYLE_BOTTLENECK];
	static int32_t s_cnt = 0;
	if (s_cnt++ % INTERVAL_TO_CALCULATE_CONTENT_BOTTLENECK == 0 || s_styleBottleneckUpdated) {
		constexpr float ratio = 0.5f;
		StylePredictionEngine::RESULT stylePredictionResult;
		s_stylePredictionEngine->invoke(originalMat, stylePredictionResult);
		for (int32_t i = 0; i < StylePredictionEngine::SIZE_STYLE_BOTTLENECK; i++) {
			s_mergedStyleBottleneck[i] = ratio * stylePredictionResult.styleBottleneck[i] + (1 - ratio) * s_styleBottleneck[i];
		}
	}

	StyleTransferEngine::RESULT styleTransferResult;
	s_styleTransferEngine->invoke(originalMat, s_mergedStyleBottleneck, StylePredictionEngine::SIZE_STYLE_BOTTLENECK, styleTransferResult);

	/* Return the results */
	originalMat = styleTransferResult.image;
	outputParam->timePreProcess = styleTransferResult.timePreProcess;
	outputParam->timeInference = styleTransferResult.timeInference;
	outputParam->timePostProcess = styleTransferResult.timePostProcess;

	return 0;
}

