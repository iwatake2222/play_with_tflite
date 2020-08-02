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

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
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
static PalmDetection palmDetection;
static HandLandmark handLandmark;
static RECT s_palmByLm;
static bool s_isPalmByLmValid = false;

/*** Function ***/
static void calcAverageRect(RECT &rectOrg, HandLandmark::HAND_LANDMARK &rectNew, float ratio);


int ImageProcessor_initialize(const INPUT_PARAM *inputParam)
{
	palmDetection.initialize(inputParam->workDir, inputParam->numThreads);
	handLandmark.initialize(inputParam->workDir, inputParam->numThreads);

	return 0;
}

int ImageProcessor_finalize(void)
{
	palmDetection.finalize();
	handLandmark.finalize();
	return 0;
}


int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	bool isPalmValid = false;
	RECT palm = { 0 };
	if (s_isPalmByLmValid == true) {
		/* Use the estimated palm position from the previous frame */
		isPalmValid = true;
		palm.x = s_palmByLm.x;
		palm.y = s_palmByLm.y;
		palm.width = s_palmByLm.width;
		palm.height = s_palmByLm.height;
		palm.rotation = s_palmByLm.rotation;
	} else {
		/*** Get Palms ***/
		std::vector<PalmDetection::PALM> palmList;
		palmDetection.invoke(*mat, palmList);
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
	}
	palm = palm.fix(mat->cols, mat->rows);

	/*** Get landmark ***/
	if (isPalmValid) {
		cv::Scalar colorRect = (s_isPalmByLmValid) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
		cv::rectangle(*mat, cv::Rect(palm.x, palm.y, palm.width, palm.height), colorRect, 3);

		/* Get landmark */
		HandLandmark::HAND_LANDMARK landmark;
		handLandmark.invoke(*mat, landmark, palm.x, palm.y, palm.width, palm.height, palm.rotation);

		if (landmark.handflag > 0.5) {
			calcAverageRect(s_palmByLm, landmark, 0.4f);
			cv::rectangle(*mat, cv::Rect(s_palmByLm.x, s_palmByLm.y, s_palmByLm.width, s_palmByLm.height), cv::Scalar(255, 0, 0), 3);

			/* Display hand landmark */
			for (int i = 0; i < 21; i++) {
				cv::circle(*mat, cv::Point(landmark.pos[i].x, landmark.pos[i].y), 3, cv::Scalar(255, 255, 0), 1);
				cv::putText(*mat, std::to_string(i), cv::Point(landmark.pos[i].x - 10, landmark.pos[i].y - 10), 1, 1, cv::Scalar(255, 255, 0));
			}
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 3; j++) {
					int indexStart = 4 * i + 1 + j;
					int indexEnd = indexStart + 1;
					int color = std::min((int)std::max((landmark.pos[indexStart].z + landmark.pos[indexEnd].z) / 2.0f * -4, 0.f), 255);
					cv::line(*mat, cv::Point(landmark.pos[indexStart].x, landmark.pos[indexStart].y), cv::Point(landmark.pos[indexEnd].x, landmark.pos[indexEnd].y), cv::Scalar(color, color, color), 3);
				}
			}

			s_isPalmByLmValid = true;
		} else {
			s_isPalmByLmValid = false;
		}
	}

	return 0;
}


static void calcAverageRect(RECT &rectOrg, HandLandmark::HAND_LANDMARK &rectNew, float ratio)
{
	if (rectOrg.width == 0) ratio = 1;	// for the first time
	rectOrg.x = (int)(rectNew.rect.x * ratio + rectOrg.x * (1 - ratio));
	rectOrg.y = (int)(rectNew.rect.y * ratio + rectOrg.y * (1 - ratio));
	rectOrg.width = (int)(rectNew.rect.width * ratio + rectOrg.width * (1 - ratio));
	rectOrg.height = (int)(rectNew.rect.height * ratio + rectOrg.height * (1 - ratio));
	rectOrg.rotation = rectNew.rect.rotation * ratio + rectOrg.rotation * (1 - ratio);
}

