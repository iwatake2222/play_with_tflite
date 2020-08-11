/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "PalmDetection.h"
#include "HandLandmark.h"
#include "ImageProcessor.h"

/*** Setting ***/
#define INTERVAL_TO_ENFORCE_PALM_DET 5
#define INTERVAL_TO_SKIP_PALM_DET 2

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, "[ImageProcessor] " __VA_ARGS__)
#else
#define PRINT(fmt, ...) printf("[ImageProcessor] " fmt, __VA_ARGS__)
#endif

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

class RECT {
public:
	int x;
	int y;
	int width;
	int height;
	float rotation;
	RECT fix(int imageWidth, int imageHeight) {
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
static int s_frameCnt;
static PalmDetection s_palmDetection;
static HandLandmark s_handLandmark;
static RECT s_palmByLm;
static bool s_isPalmByLmValid = false;

/*** Function ***/
static cv::Scalar convertColorBgrToAppropreate(cv::Scalar color) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(color[2], color[1], color[0]);
#else
	return color;
#endif
}

static void calcAverageRect(RECT &rectOrg, HandLandmark::HAND_LANDMARK &rectNew, float ratioPos, float ratioSize)
{
	if (rectOrg.width == 0) {
		// for the first time
		ratioPos = 1;
		ratioSize = 1;
	}
	rectOrg.x = (int)(rectNew.rect.x * ratioPos + rectOrg.x * (1 - ratioPos));
	rectOrg.y = (int)(rectNew.rect.y * ratioPos + rectOrg.y * (1 - ratioPos));
	rectOrg.width = (int)(rectNew.rect.width * ratioSize + rectOrg.width * (1 - ratioSize));
	rectOrg.height = (int)(rectNew.rect.height * ratioSize + rectOrg.height * (1 - ratioSize));
	rectOrg.rotation = rectNew.rect.rotation * ratioSize + rectOrg.rotation * (1 - ratioSize);
}

int ImageProcessor_initialize(const INPUT_PARAM *inputParam)
{
	s_palmDetection.initialize(inputParam->workDir, inputParam->numThreads);
	s_handLandmark.initialize(inputParam->workDir, inputParam->numThreads);

	return 0;
}

int ImageProcessor_finalize(void)
{
	s_palmDetection.finalize();
	s_handLandmark.finalize();
	return 0;
}

int ImageProcessor_command(int cmd)
{
	switch (cmd) {
	default:
		PRINT("command(%d) is not supported\n", cmd);
		return -1;
	}
}

int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	s_frameCnt++;
	bool enforcePalmDet = (s_frameCnt % INTERVAL_TO_ENFORCE_PALM_DET) == 0;	// to increase accuracy
	bool skipPalmDet = (s_frameCnt % INTERVAL_TO_SKIP_PALM_DET) != 0;			// to increase fps
	bool isPalmValid = false;
	RECT palm = { 0 };
	if (s_isPalmByLmValid == false || (enforcePalmDet && !skipPalmDet)) {
		/*** Get Palms ***/
		std::vector<PalmDetection::PALM> palmList;
		if (!skipPalmDet) {
			s_palmDetection.invoke(*mat, palmList);
		}
		for (const auto detPalm : palmList) {
			s_palmByLm.width = 0;	// reset 
			palm.x = (int)(detPalm.x * 1);
			palm.y = (int)(detPalm.y * 1);
			palm.width = (int)(detPalm.width * 1);
			palm.height = (int)(detPalm.height * 1);
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
	palm = palm.fix(mat->cols, mat->rows);

	/*** Get landmark ***/
	HandLandmark::HAND_LANDMARK landmark = { 0 };
	if (isPalmValid) {
		cv::Scalar colorRect = (s_isPalmByLmValid) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
		cv::rectangle(*mat, cv::Rect(palm.x, palm.y, palm.width, palm.height), convertColorBgrToAppropreate(colorRect), 3);

		/* Get landmark */
		s_handLandmark.invoke(*mat, landmark, palm.x, palm.y, palm.width, palm.height, palm.rotation);

		if (landmark.handflag >= 0.8) {
			calcAverageRect(s_palmByLm, landmark, 0.6f, 0.4f);
			cv::rectangle(*mat, cv::Rect(s_palmByLm.x, s_palmByLm.y, s_palmByLm.width, s_palmByLm.height), convertColorBgrToAppropreate(cv::Scalar(255, 0, 0)), 3);

			/* Display hand landmark */
			for (int i = 0; i < 21; i++) {
				cv::circle(*mat, cv::Point((int)landmark.pos[i].x, (int)landmark.pos[i].y), 3, convertColorBgrToAppropreate(cv::Scalar(255, 255, 0)), 1);
				cv::putText(*mat, std::to_string(i), cv::Point((int)landmark.pos[i].x - 10, (int)landmark.pos[i].y - 10), 1, 1, convertColorBgrToAppropreate(cv::Scalar(255, 255, 0)));
			}
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 3; j++) {
					int indexStart = 4 * i + 1 + j;
					int indexEnd = indexStart + 1;
					int color = std::min((int)std::max((landmark.pos[indexStart].z + landmark.pos[indexEnd].z) / 2.0f * -4, 0.f), 255);
					cv::line(*mat, cv::Point((int)landmark.pos[indexStart].x, (int)landmark.pos[indexStart].y), cv::Point((int)landmark.pos[indexEnd].x, (int)landmark.pos[indexEnd].y), convertColorBgrToAppropreate(cv::Scalar(color, color, color)), 3);
				}
			}
			s_isPalmByLmValid = true;
		} else {
			s_isPalmByLmValid = false;
		}
	}

	return 0;
}


