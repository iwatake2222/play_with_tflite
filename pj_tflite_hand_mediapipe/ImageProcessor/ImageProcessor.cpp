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
#include "PalmDetectionEngine.h"
#include "HandLandmarkEngine.h"
#include "ImageProcessor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Setting ***/
#define INTERVAL_TO_ENFORCE_PALM_DET 5

class RECT {
public:
	int32_t x;
	int32_t y;
	int32_t width;
	int32_t height;
	float rotation;
	RECT fix(int32_t imageWidth, int32_t imageHeight) {
		RECT rect;
		rect.x = std::max(0, std::min(imageWidth, x));
		rect.y = std::max(0, std::min(imageHeight, y));
		rect.width = std::max(0, std::min(imageWidth - x, width));
		rect.height = std::max(0, std::min(imageHeight - y, height));
		rect.rotation = rotation;
		return rect;
	}
};

/*** Global variable ***/
static std::unique_ptr<PalmDetectionEngine> s_palmDetection;
static std::unique_ptr<HandLandmarkEngine> s_handLandmark;
static int32_t s_frameCnt;
static RECT s_palmByLm;
static bool s_isPalmByLmValid = false;



/*** Function ***/
static void calcAverageRect(RECT &rectOrg, HandLandmarkEngine::HAND_LANDMARK &rectNew, float ratioPos, float ratioSize);

static inline cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam)
{
	if (s_palmDetection || s_handLandmark) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_palmDetection.reset(new PalmDetectionEngine());
	if (s_palmDetection->initialize(inputParam->workDir, inputParam->numThreads) != PalmDetectionEngine::RET_OK) {
		return -1;
	}
	s_handLandmark.reset(new HandLandmarkEngine());
	if (s_handLandmark->initialize(inputParam->workDir, inputParam->numThreads) != HandLandmarkEngine::RET_OK) {
		return -1;
	}
	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_palmDetection || !s_handLandmark) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_palmDetection->finalize() != PalmDetectionEngine::RET_OK) {
		return -1;
	}
	if (s_handLandmark->finalize() != HandLandmarkEngine::RET_OK) {
		return -1;
	}
	s_palmDetection.reset();
	s_handLandmark.reset();

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_palmDetection || !s_handLandmark) {
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
	if (!s_palmDetection || !s_handLandmark) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	s_frameCnt++;
	cv::Mat& originalMat = *mat;
	
	//bool enforcePalmDet = (s_frameCnt % INTERVAL_TO_ENFORCE_PALM_DET) == 0;		// to increase accuracy
	bool enforcePalmDet = false;
	bool isPalmValid = false;
	PalmDetectionEngine::RESULT palmResult;
	RECT palm = { 0 };
	if (s_isPalmByLmValid == false || enforcePalmDet) {
		/*** Get Palms ***/
		s_palmDetection->invoke(originalMat, palmResult);
		for (const auto& detPalm : palmResult.palmList) {
			s_palmByLm.width = 0;	// reset 
			palm.x = (int32_t)(detPalm.x * 1);
			palm.y = (int32_t)(detPalm.y * 1);
			palm.width = (int32_t)(detPalm.width * 1);
			palm.height = (int32_t)(detPalm.height * 1);
			palm.rotation = detPalm.rotation;
			isPalmValid = true;
			break;	// use only one palm
		}
	} else {
		/* Use the estimated palm position from the previous frame */
		isPalmValid = true;
		palm.x = s_palmByLm.x;
		palm.y = s_palmByLm.y;
		palm.width = s_palmByLm.width;
		palm.height = s_palmByLm.height;
		palm.rotation = s_palmByLm.rotation;
	}
	palm = palm.fix(originalMat.cols, originalMat.rows);

	/*** Get landmark ***/
	HandLandmarkEngine::RESULT landmarkResult;
	if (isPalmValid) {
		cv::Scalar colorRect = (s_isPalmByLmValid) ? createCvColor(0, 255, 0) : createCvColor(0, 0, 255);
		cv::rectangle(originalMat, cv::Rect(palm.x, palm.y, palm.width, palm.height), colorRect, 3);

		/* Get landmark */
		s_handLandmark->invoke(originalMat, palm.x, palm.y, palm.width, palm.height, palm.rotation, landmarkResult);

		if (landmarkResult.handLandmark.handflag >= 0.8) {
			calcAverageRect(s_palmByLm, landmarkResult.handLandmark, 0.6f, 0.4f);
			cv::rectangle(originalMat, cv::Rect(s_palmByLm.x, s_palmByLm.y, s_palmByLm.width, s_palmByLm.height), createCvColor(255, 0, 0), 3);

			/* Display hand landmark */
			for (int32_t i = 0; i < 21; i++) {
				cv::circle(originalMat, cv::Point((int32_t)landmarkResult.handLandmark.pos[i].x, (int32_t)landmarkResult.handLandmark.pos[i].y), 3, createCvColor(255, 255, 0), 1);
				cv::putText(originalMat, std::to_string(i), cv::Point((int32_t)landmarkResult.handLandmark.pos[i].x - 10, (int32_t)landmarkResult.handLandmark.pos[i].y - 10), 1, 1, createCvColor(255, 255, 0));
			}
			for (int32_t i = 0; i < 5; i++) {
				for (int32_t j = 0; j < 3; j++) {
					int32_t indexStart = 4 * i + 1 + j;
					int32_t indexEnd = indexStart + 1;
					int32_t color = std::min((int32_t)std::max((landmarkResult.handLandmark.pos[indexStart].z + landmarkResult.handLandmark.pos[indexEnd].z) / 2.0f * -4, 0.f), 255);
					cv::line(originalMat, cv::Point((int32_t)landmarkResult.handLandmark.pos[indexStart].x, (int32_t)landmarkResult.handLandmark.pos[indexStart].y), cv::Point((int32_t)landmarkResult.handLandmark.pos[indexEnd].x, (int32_t)landmarkResult.handLandmark.pos[indexEnd].y), createCvColor(color, color, color), 3);
				}
			}
			s_isPalmByLmValid = true;
		} else {
			s_isPalmByLmValid = false;
		}
	}

	/* Return the results */
	outputParam->timePreProcess = palmResult.timePreProcess + landmarkResult.timePreProcess;
	outputParam->timeInference = palmResult.timeInference + landmarkResult.timeInference;
	outputParam->timePostProcess = palmResult.timePostProcess  + landmarkResult.timePostProcess;

	return 0;
}

static void calcAverageRect(RECT &rectOrg, HandLandmarkEngine::HAND_LANDMARK &rectNew, float ratioPos, float ratioSize)
{
	if (rectOrg.width == 0) {
		// for the first time
		ratioPos = 1;
		ratioSize = 1;
	}
	rectOrg.x = (int32_t)(rectNew.rect.x * ratioPos + rectOrg.x * (1 - ratioPos));
	rectOrg.y = (int32_t)(rectNew.rect.y * ratioPos + rectOrg.y * (1 - ratioPos));
	rectOrg.width = (int32_t)(rectNew.rect.width * ratioSize + rectOrg.width * (1 - ratioSize));
	rectOrg.height = (int32_t)(rectNew.rect.height * ratioSize + rectOrg.height * (1 - ratioSize));
	rectOrg.rotation = rectNew.rect.rotation * ratioSize + rectOrg.rotation * (1 - ratioSize);
}

