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
#include <opencv2/tracking/tracker.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "PalmDetectionEngine.h"
#include "HandLandmarkEngine.h"
#include "ClassificationEngine.h"
#include "AreaSelector.h"
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

typedef struct {
	cv::Ptr<cv::Tracker> tracker;
	int32_t numLost;
	std::string labelName;
	RECT rectFirst;
} OBJECT_TRACKER;

/*** Global variable ***/
static std::unique_ptr<PalmDetectionEngine> s_palmDetectionEngine;
static std::unique_ptr<HandLandmarkEngine> s_handLandmarkEngine;
static std::unique_ptr<ClassificationEngine> s_classificationEngine;
AreaSelector s_areaSelector;
static int32_t s_frameCnt;
static RECT s_palmByLm;
static bool s_isPalmByLmValid = false;
static std::vector<OBJECT_TRACKER> s_objectList;
static int32_t s_animCount = 0;
static bool s_isDebug = true;


/*** Function ***/
static void calcAverageRect(RECT &rectOrg, HandLandmarkEngine::HAND_LANDMARK &rectNew, float ratioPos, float ratioSize);

static inline cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

static inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
	cv::Ptr<cv::Tracker> tracker;

	if (name == "KCF")
		tracker = cv::TrackerKCF::create();
	else if (name == "TLD")
		tracker = cv::TrackerTLD::create();
	else if (name == "BOOSTING")
		tracker = cv::TrackerBoosting::create();
	else if (name == "MEDIAN_FLOW")
		tracker = cv::TrackerMedianFlow::create();
	else if (name == "MIL")
		tracker = cv::TrackerMIL::create();
	else if (name == "GOTURN")
		tracker = cv::TrackerGOTURN::create();
	else if (name == "MOSSE")
		tracker = cv::TrackerMOSSE::create();
	else
		CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

	return tracker;
}

static void drawRing(cv::Mat &mat, RECT rect, cv::Scalar color, int animCount)
{
	/* Reference: https://github.com/Kazuhito00/object-detection-bbox-art */
	animCount *= int(135 / 30.0);
	cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
	cv::Point radius((rect.width + rect.height) / 4, (rect.width + rect.height) / 4);
	int  ring_thickness = std::max(int(radius.x / 20), 1);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 80 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 150 + animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 200 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 230 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 260 + animCount, 0, 60, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 337 + animCount, 0, 5, color, ring_thickness);

	radius *= 0.9;
	ring_thickness = std::max(int(radius.x / 12), 1);
	cv::ellipse(mat, cv::Point(center), radius, 0 - animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 80 - animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 150 - animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 200 - animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 260 - animCount, 0, 60, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 3370 - animCount, 0, 5, color, ring_thickness);

	radius *= 0.9;
	ring_thickness = std::max(int(radius.x / 15), 1);
	cv::ellipse(mat, cv::Point(center), radius, 30 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 110 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 180 + animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 230 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 260 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 290 + animCount, 0, 60, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 367 + animCount, 0, 5, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
}

static void drawText(cv::Mat &mat, RECT rect, cv::Scalar color, std::string str, int animCount)
{
	/* Reference: https://github.com/Kazuhito00/object-detection-bbox-art */
	double font_size = std::min((rect.width + rect.height) / 2 * 0.1, 1.0);
	cv::Point drawpoint1(rect.x + rect.width / 2, rect.y + rect.height / 2);
	cv::Point drawpoint2(rect.x + rect.width - rect.width / 10, rect.y + int(rect.height / 10));
	cv::Point drawpoint3(rect.x + rect.width + int(rect.width / 2), rect.y + int(rect.height / 10));
	cv::Point textpoint(drawpoint2.x, drawpoint2.y - int(font_size * 2.0));
	if (drawpoint3.x > mat.cols) {
		drawpoint2 = cv::Point(rect.x + rect.width / 10, rect.y + int(rect.height / 10));
		drawpoint3 = cv::Point(rect.x - rect.width / 2, rect.y + int(rect.height / 10));
		textpoint = cv::Point(drawpoint3.x, drawpoint2.y - int(font_size * 1.5));
	}
	cv::circle(mat, drawpoint1, int(rect.width / 40), color, -1);
	cv::line(mat, drawpoint1, drawpoint2, color, std::max(2, rect.width / 80));
	cv::line(mat, drawpoint2, drawpoint3, color, std::max(2, rect.width / 80));

	cv::putText(mat, str, textpoint, cv::FONT_HERSHEY_DUPLEX, font_size, createCvColor(171, 97, 50), 5);
	cv::putText(mat, str, textpoint, cv::FONT_HERSHEY_DUPLEX, font_size, color, 2);
}

static std::string classify(cv::Mat &originalMat, const cv::Rect &selectedArea)
{
	/* Classify the selected area */
	cv::Rect targetArea = s_areaSelector.m_selectedArea;
	const int centerX = targetArea.x + targetArea.width / 2;
	const int centerY = targetArea.y + targetArea.height / 2;
	int width = (int)(targetArea.width * 1.2); // expand
	int height = (int)(targetArea.height * 1.2); // expand
	targetArea.x = std::max(centerX - width / 2, 0);
	targetArea.y = std::max(centerY - height / 2, 0);
	targetArea.width = std::min(width, originalMat.cols - targetArea.x);
	targetArea.height = std::min(height, originalMat.rows - targetArea.y);
	cv::Mat targetImage = originalMat(targetArea);
	ClassificationEngine::RESULT resultWithoutPadding;
	s_classificationEngine->invoke(targetImage, resultWithoutPadding);

	width = std::max(targetArea.width, targetArea.height);
	height = std::max(targetArea.width, targetArea.height);
	targetArea.x = std::max(centerX - width / 2, 0);
	targetArea.y = std::max(centerY - height / 2, 0);
	targetArea.width = std::min(width, originalMat.cols - targetArea.x);
	targetArea.height = std::min(height, originalMat.rows - targetArea.y);
	cv::Mat targetImageWithPadding = originalMat(targetArea);

	ClassificationEngine::RESULT resultWithPadding;
	s_classificationEngine->invoke(targetImageWithPadding, resultWithPadding);

	return (resultWithoutPadding.score > resultWithPadding.score) ? resultWithoutPadding.labelName : resultWithPadding.labelName;
}

int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam)
{
	if (s_palmDetectionEngine || s_handLandmarkEngine || s_classificationEngine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_palmDetectionEngine.reset(new PalmDetectionEngine());
	if (s_palmDetectionEngine->initialize(inputParam->workDir, inputParam->numThreads) != PalmDetectionEngine::RET_OK) {
		return -1;
	}
	s_handLandmarkEngine.reset(new HandLandmarkEngine());
	if (s_handLandmarkEngine->initialize(inputParam->workDir, inputParam->numThreads) != HandLandmarkEngine::RET_OK) {
		return -1;
	}
	s_classificationEngine.reset(new ClassificationEngine());
	if (s_classificationEngine->initialize(inputParam->workDir, inputParam->numThreads) != HandLandmarkEngine::RET_OK) {
		return -1;
	}

	cv::setNumThreads(inputParam->numThreads);

	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_palmDetectionEngine || !s_handLandmarkEngine || !s_classificationEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_palmDetectionEngine->finalize() != PalmDetectionEngine::RET_OK) {
		return -1;
	}
	if (s_handLandmarkEngine->finalize() != HandLandmarkEngine::RET_OK) {
		return -1;
	}
	if (s_classificationEngine->finalize() != HandLandmarkEngine::RET_OK) {
		return -1;
	}
	s_palmDetectionEngine.reset();
	s_handLandmarkEngine.reset();
	s_classificationEngine.reset();

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_palmDetectionEngine || !s_handLandmarkEngine || !s_classificationEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	switch (cmd) {
	case 0:
		s_isDebug = !s_isDebug;
		return 0;
		break;
	default:
		PRINT_E("command(%d) is not supported\n", cmd);
		return -1;
	}
}


int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam)
{
	if (!s_palmDetectionEngine || !s_handLandmarkEngine || !s_classificationEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	s_frameCnt++;
	cv::Mat& originalMat = *mat;

	bool enforcePalmDet = (s_frameCnt % INTERVAL_TO_ENFORCE_PALM_DET) == 0;		// to increase accuracy
	//bool enforcePalmDet = false;
	bool isPalmValid = false;
	PalmDetectionEngine::RESULT palmResult;
	RECT palm = { 0 };
	if (s_isPalmByLmValid == false || enforcePalmDet) {
		/*** Get Palms ***/
		s_palmDetectionEngine->invoke(originalMat, palmResult);
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
		s_handLandmarkEngine->invoke(originalMat, palm.x, palm.y, palm.width, palm.height, palm.rotation, landmarkResult);

		if (landmarkResult.handLandmark.handflag >= 0.8) {
			calcAverageRect(s_palmByLm, landmarkResult.handLandmark, 0.6f, 0.4f);
			cv::rectangle(originalMat, cv::Rect(s_palmByLm.x, s_palmByLm.y, s_palmByLm.width, s_palmByLm.height), createCvColor(255, 0, 0), 3);

			/* Display hand landmark */
			if (s_isDebug) {
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
			} else {
				for (int i = 0; i < 21; i++) {
					cv::circle(*mat, cv::Point((int)landmarkResult.handLandmark.pos[i].x, (int)landmarkResult.handLandmark.pos[i].y), 3, createCvColor(255, 255, 0), 1);
				}
			}
			s_isPalmByLmValid = true;
		} else {
			s_isPalmByLmValid = false;
		}
	}

	/* Select area according to finger pose and position */
	s_areaSelector.run(landmarkResult.handLandmark);
	PRINT("areaSelector.m_status = %d \n", s_areaSelector.m_status);
	if (landmarkResult.handLandmark.handflag >= 0.8) {
		s_areaSelector.m_selectedArea.x = std::min(std::max(0, s_areaSelector.m_selectedArea.x), mat->cols);
		s_areaSelector.m_selectedArea.y = std::min(std::max(0, s_areaSelector.m_selectedArea.y), mat->rows);
		s_areaSelector.m_selectedArea.width = std::min(std::max(1, s_areaSelector.m_selectedArea.width), mat->cols - s_areaSelector.m_selectedArea.x);
		s_areaSelector.m_selectedArea.height = std::min(std::max(1, s_areaSelector.m_selectedArea.height), mat->rows - s_areaSelector.m_selectedArea.y);
		switch (s_areaSelector.m_status) {
		case AreaSelector::STATUS_AREA_SELECT_INIT:
			cv::putText(*mat, "Point index and middle fingers at the start point", cv::Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 0.8, createCvColor(0, 255, 0), 2);
			break;
		case AreaSelector::STATUS_AREA_SELECT_DRAG:
			cv::putText(*mat, "Move the fingers to the end point,", cv::Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 0.8, createCvColor(0, 255, 0), 2);
			cv::putText(*mat, "then put back the middle finger", cv::Point(0, 40), cv::FONT_HERSHEY_DUPLEX, 0.8, createCvColor(0, 255, 0), 2);
			cv::rectangle(*mat, s_areaSelector.m_selectedArea, createCvColor(255, 0, 0));
			break;
		case AreaSelector::STATUS_AREA_SELECT_SELECTED:
		{
			std::string labelName = classify(*mat, s_areaSelector.m_selectedArea);

			/* Add a new tracker for the selected area */
			OBJECT_TRACKER object;
			if (s_areaSelector.m_selectedArea.width * s_areaSelector.m_selectedArea.height > mat->cols * mat->rows * 0.1) {
				object.tracker = createTrackerByName("MEDIAN_FLOW");	// Use median flow for hube object because KCF becomes slow when the object size is huge
			} else {
				object.tracker = createTrackerByName("KCF");
			}
			object.tracker->init(*mat, cv::Rect(s_areaSelector.m_selectedArea));
			object.numLost = 0;
			object.labelName = labelName;
			//object.rectFirst = s_selectedArea;
			s_objectList.push_back(object);
		}
		break;
		default:
			break;
		}
	}

	/* Track and display tracked objects */

	s_animCount++;
	for (auto it = s_objectList.begin(); it != s_objectList.end();) {
		auto tracker = it->tracker;
		cv::Rect2d trackedRect;
		if (tracker->update(*mat, trackedRect)) {
			//cv::rectangle(mat, trackedRect, cv::Scalar(255, 0, 0), 2, 1);
			RECT rect;
			rect.x = (int)trackedRect.x;
			rect.y = (int)trackedRect.y;
			rect.width = (int)trackedRect.width;
			rect.height = (int)trackedRect.height;
			drawRing(*mat, rect, createCvColor(255, 255, 205), s_animCount);
			drawText(*mat, rect, createCvColor(207, 161, 69), it->labelName, s_animCount);
			it->numLost = 0;
			if (rect.width > mat->cols * 0.9) {	// in case median flow outputs crazy result
				PRINT("delete due to too big result\n");
				it = s_objectList.erase(it);
				tracker.release();
			} else {
				it++;
			}
		} else {
			PRINT("lost\n");
			if (++(it->numLost) > 20) {
				PRINT("delete\n");
				it = s_objectList.erase(it);
				tracker.release();
			} else {
				it++;
			}
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

