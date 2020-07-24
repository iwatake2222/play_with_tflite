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

typedef struct {
	int x;
	int y;
	int width;
	int height;
	float rotation;
} RECT;

/*** Global variable ***/
static PalmDetection palmDetection;
static HandLandmark handLandmark;
static RECT s_palmByLm;
static bool s_isPalmByLmValid = false;

/*** Function ***/
static void calcAverageRect(RECT &rectOrg, RECT &rectNew, float ratio);
static int transformLandmarkToRect(const HandLandmark::HAND_LANDMARK &landmark, RECT *rect);


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
	int palmX = s_palmByLm.x;
	int palmY = s_palmByLm.y;
	int palmW = s_palmByLm.width;
	int palmH = s_palmByLm.height;
	float palmRotation = s_palmByLm.rotation;

	if (s_isPalmByLmValid == true) {
		isPalmValid = true;
	} else {
		/*** Get Palms ***/
		std::vector<PalmDetection::PALM> palmList;
		palmDetection.invoke(*mat, palmList);
		for (const auto palm : palmList) {
			s_palmByLm.width = 0;	// reset 
			palmX = (int)(palm.x * mat->cols);
			palmY = (int)(palm.y * mat->rows);
			palmW = (int)(palm.width * mat->cols);
			palmH = (int)(palm.height * mat->rows);
			palmRotation = palm.rotation;
			isPalmValid = true;
			break;	// use only one palm
		}
	}

	/*** Get landmark ***/
	if (isPalmValid) {
		cv::Scalar colorRect = (s_isPalmByLmValid) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
		cv::rectangle(*mat, cv::Rect(palmX, palmY, palmW, palmH), colorRect, 3);

		/* Rotate palm image */
		cv::Mat matForLM;
		cv::RotatedRect rect(cv::Point(palmX + palmW / 2, palmY + palmH / 2), cv::Size(palmW, palmH), palmRotation * 180.f / 3.141592654f);
		cv::Mat trans = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0);
		cv::Mat srcRot;
		cv::warpAffine(*mat, srcRot, trans, mat->size());
		cv::getRectSubPix(srcRot, rect.size, rect.center, matForLM);

		/* Get landmark */
		HandLandmark::HAND_LANDMARK landmark;
		handLandmark.invoke(matForLM, landmark);

		if (landmark.handflag > 0.5) {

			/* Fix landmark rotation */
			for (int i = 0; i < 21; i++) {
				landmark.pos[i].x *= matForLM.cols;
				landmark.pos[i].y *= matForLM.rows;
			}
			handLandmark.rotateLandmark(landmark, palmRotation, matForLM.cols, matForLM.rows);

			for (int i = 0; i < 21; i++) {
				landmark.pos[i].x += palmX;
				landmark.pos[i].y += palmY;
			}

			/* Display hand landmark */
			for (int i = 0; i < 21; i++) {
				cv::circle(*mat, cv::Point(landmark.pos[i].x, landmark.pos[i].y), 3, cv::Scalar(255, 255, 0), 1);
				cv::putText(*mat, std::to_string(i), cv::Point(landmark.pos[i].x - 10, landmark.pos[i].y - 10), 1, 1, cv::Scalar(255, 255, 0));
			}
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 3; j++) {
					int indexStart = 4 * i + 1 + j;
					int indexEnd = indexStart + 1;
					int color = std::min((int)std::max(-landmark.pos[indexStart].z * 255, 0.f), 255);
					cv::line(*mat, cv::Point(landmark.pos[indexStart].x, landmark.pos[indexStart].y), cv::Point(landmark.pos[indexEnd].x, landmark.pos[indexEnd].y), cv::Scalar(color, color, color), 3);
				}
			}

			/* Calculate palm rectangle for the next time */
			RECT palm;
			transformLandmarkToRect(landmark, &palm);
			palm.x = std::max(0, palm.x);
			palm.y = std::max(0, palm.y);
			palm.width = std::min(mat->cols - palm.x, palm.width);
			palm.height = std::min(mat->rows - palm.y, palm.height);
			palm.rotation = handLandmark.calculateRotation(landmark);
			calcAverageRect(s_palmByLm, palm, 0.4f);
			
			cv::rectangle(*mat, cv::Rect(s_palmByLm.x, s_palmByLm.y, s_palmByLm.width, s_palmByLm.height), cv::Scalar(255, 0, 0), 3);

			s_isPalmByLmValid = true;
		} else {
			s_isPalmByLmValid = false;
		}

	}

	return 0;
}



static int transformLandmarkToRect(const HandLandmark::HAND_LANDMARK &landmark, RECT *rect)
{
	const float shift_x = 0.0f;
	const float shift_y = -0.0f;
	const float scale_x = 1.8f;
	const float scale_y = 1.8f;

	float width = 0;
	float height = 0;
	float x_center = 0;
	float y_center = 0;

	float xmin = landmark.pos[0].x;
	float xmax = landmark.pos[0].x;
	float ymin = landmark.pos[0].y;
	float ymax = landmark.pos[0].y;

	for (int i = 0; i < 21; i++) {
		if (landmark.pos[i].x < xmin) xmin = landmark.pos[i].x;
		if (landmark.pos[i].x > xmax) xmax = landmark.pos[i].x;
		if (landmark.pos[i].y < ymin) ymin = landmark.pos[i].y;
		if (landmark.pos[i].y > ymax) ymax = landmark.pos[i].y;
	}
	width = xmax - xmin;
	height = ymax - ymin;
	x_center = (xmax + xmin) / 2.f;
	y_center = (ymax + ymin) / 2.f;

	width *= scale_x;
	height *= scale_y;

	const float long_side = std::max(width, height);
	rect->width = (int)(long_side * 1);
	rect->height = (int)(long_side * 1);
	rect->x = (int)(x_center - rect->width / 2);
	rect->y = (int)(y_center - rect->height / 2);

	return 0;
}



static void calcAverageRect(RECT &rectOrg, RECT &rectNew, float ratio)
{
	if (rectOrg.width == 0) ratio = 1;	// for the first time
	rectOrg.x = (int)(rectNew.x * ratio + rectOrg.x * (1 - ratio));
	rectOrg.y = (int)(rectNew.y * ratio + rectOrg.y * (1 - ratio));
	rectOrg.width = (int)(rectNew.width * ratio + rectOrg.width * (1 - ratio));
	rectOrg.height = (int)(rectNew.height * ratio + rectOrg.height * (1 - ratio));
	rectOrg.rotation = (int)(rectNew.rotation * ratio + rectOrg.rotation * (1 - ratio));
}

