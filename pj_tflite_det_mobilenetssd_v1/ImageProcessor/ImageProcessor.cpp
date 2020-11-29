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
#include "DetectionEngine.h"
#include "ImageProcessor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<DetectionEngine> s_classificationEngine;

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
	if (s_classificationEngine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_classificationEngine.reset(new DetectionEngine());
	if (s_classificationEngine->initialize(inputParam->workDir, inputParam->numThreads) != DetectionEngine::RET_OK) {
		s_classificationEngine->finalize();
		s_classificationEngine.reset();
		return -1;
	}
	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_classificationEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_classificationEngine->finalize() != DetectionEngine::RET_OK) {
		return -1;
	}

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_classificationEngine) {
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


int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam)
{
	if (!s_classificationEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	const cv::Mat originalMat = *mat;
	DetectionEngine::RESULT result;
	result.objectList.clear();
	if (s_classificationEngine->invoke(originalMat, result) != DetectionEngine::RET_OK) {
		return -1;
	}

	/* Draw the result */
	for (const auto& object : result.objectList) {
		cv::rectangle(originalMat, cv::Rect(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y), static_cast<int32_t>(object.width), static_cast<int32_t>(object.height)), cv::Scalar(255, 255, 0), 3);
		cv::putText(originalMat, object.label, cv::Point(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y) + 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 0, 0), 3);
		cv::putText(originalMat, object.label, cv::Point(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y) + 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 255, 0), 1);
	}


	/* Return the results */
	int32_t objectNum = 0;
	for (const auto& object : result.objectList) {
		outputParam->objectList[objectNum].classId = object.classId;
		snprintf(outputParam->objectList[objectNum].label, sizeof(outputParam->objectList[objectNum].label), "%s", object.label.c_str());
		outputParam->objectList[objectNum].score = object.score;
		outputParam->objectList[objectNum].x = static_cast<int32_t>(object.x);
		outputParam->objectList[objectNum].y = static_cast<int32_t>(object.y);
		outputParam->objectList[objectNum].width = static_cast<int32_t>(object.width);
		outputParam->objectList[objectNum].height = static_cast<int32_t>(object.height);
		objectNum++;
		if (objectNum >= NUM_MAX_RESULT) break;
	}
	outputParam->objectNum = objectNum;
	outputParam->timePreProcess = result.timePreProcess;
	outputParam->timeInference = result.timeInference;
	outputParam->timePostProcess = result.timePostProcess;

	return 0;
}

