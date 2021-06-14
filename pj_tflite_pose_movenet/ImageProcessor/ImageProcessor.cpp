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
#include "CommonHelper.h"
#include "PoseEngine.h"
#include "ImageProcessor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<PoseEngine> s_poseEngine;

/*** Function ***/
static cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}


int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam)
{
	if (s_poseEngine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_poseEngine.reset(new PoseEngine());
	if (s_poseEngine->initialize(inputParam->workDir, inputParam->numThreads) != PoseEngine::RET_OK) {
		return -1;
	}
	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_poseEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_poseEngine->finalize() != PoseEngine::RET_OK) {
		return -1;
	}

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_poseEngine) {
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

int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam)
{
	if (!s_poseEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	cv::Mat& originalMat = *mat;
	PoseEngine::RESULT result;
	if (s_poseEngine->invoke(originalMat, result) != PoseEngine::RET_OK) {
		return -1;
	}

	/* Draw the result */
	/* note: we have only one body with this model */
	constexpr float scoreThreshold = 0.2;
	const auto& scoreList = result.poseKeypointScores[0];
	const auto& partList = result.poseKeypointCoords[0];
	int32_t partNum = partList.size();

	for (const auto& jointLine : jointLineList) {
		if (scoreList[jointLine.first] >= scoreThreshold && scoreList[jointLine.second] >= scoreThreshold) {
			int32_t x0 = static_cast<int32_t>(partList[jointLine.first].first * originalMat.cols);
			int32_t y0 = static_cast<int32_t>(partList[jointLine.first].second * originalMat.rows);
			int32_t x1 = static_cast<int32_t>(partList[jointLine.second].first * originalMat.cols);
			int32_t y1 = static_cast<int32_t>(partList[jointLine.second].second * originalMat.rows);
			cv::line(originalMat, cv::Point(x0, y0), cv::Point(x1, y1) , createCvColor(200, 200, 200), 2);
		}
	}

	for (int32_t i = 0; i < partNum; i++) {
		int32_t x = static_cast<int32_t>(partList[i].first * originalMat.cols);
		int32_t y = static_cast<int32_t>(partList[i].second * originalMat.rows);
		float score = scoreList[i];
		if (score >= scoreThreshold) {
			cv::circle(originalMat, cv::Point(x, y), 5, createCvColor(0, 255, 0), -1);
		}
	}

	/* Return the results */
	outputParam->timePreProcess = result.timePreProcess;
	outputParam->timeInference = result.timeInference;
	outputParam->timePostProcess = result.timePostProcess;

	return 0;
}

