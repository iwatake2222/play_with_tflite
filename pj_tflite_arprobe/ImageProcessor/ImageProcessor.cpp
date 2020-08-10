/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include "PalmDetection.h"
#include "HandLandmark.h"
#include "AreaSelector.h"
#include "ImageProcessor.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(fmt, ...) __android_log_print(ANDROID_LOG_INFO, TAG, "[ImageProcessor] " fmt, __VA_ARGS__)
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

typedef struct {
	cv::Ptr<cv::Tracker> tracker;
	int numLost;
	RECT rectFirst;
} OBJECT_TRACKER;

/*** Global variable ***/
static PalmDetection palmDetection;
static HandLandmark handLandmark;
static RECT s_palmByLm;
static bool s_isPalmByLmValid = false;
static std::vector<OBJECT_TRACKER> s_objectList;
static AreaSelector s_areaSelector;


/*** Function ***/
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

	cv::putText(mat, str, textpoint, cv::FONT_HERSHEY_DUPLEX, font_size, cv::Scalar(171, 97, 50), 5);
	cv::putText(mat, str, textpoint, cv::FONT_HERSHEY_DUPLEX, font_size, color, 2);
}



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

		if (landmark.handflag >= 0.8) {
			calcAverageRect(s_palmByLm, landmark, 0.5f, 0.2f);
			cv::rectangle(*mat, cv::Rect(s_palmByLm.x, s_palmByLm.y, s_palmByLm.width, s_palmByLm.height), cv::Scalar(255, 0, 0), 3);

			/* Display hand landmark */
			for (int i = 0; i < 21; i++) {
				cv::circle(*mat, cv::Point((int)landmark.pos[i].x, (int)landmark.pos[i].y), 3, cv::Scalar(255, 255, 0), 1);
				cv::putText(*mat, std::to_string(i), cv::Point((int)landmark.pos[i].x - 10, (int)landmark.pos[i].y - 10), 1, 1, cv::Scalar(255, 255, 0));
			}
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 3; j++) {
					int indexStart = 4 * i + 1 + j;
					int indexEnd = indexStart + 1;
					int color = std::min((int)std::max((landmark.pos[indexStart].z + landmark.pos[indexEnd].z) / 2.0f * -4, 0.f), 255);
					cv::line(*mat, cv::Point((int)landmark.pos[indexStart].x, (int)landmark.pos[indexStart].y), cv::Point((int)landmark.pos[indexEnd].x, (int)landmark.pos[indexEnd].y), cv::Scalar(color, color, color), 3);
				}
			}

			/* Select area according to finger pose and position */
			s_areaSelector.run(landmark);
			PRINT("status = %d\n", s_areaSelector.m_status);
			if (s_areaSelector.m_status == AreaSelector::STATUS_AREA_SELECT_DRAG) {
				cv::rectangle(*mat, s_areaSelector.m_selectedArea, cv::Scalar(255, 0, 0));
			} else if (s_areaSelector.m_status == AreaSelector::STATUS_AREA_SELECT_SELECTED) {
				/* Add a new tracker for the selected area */
				OBJECT_TRACKER object;
				object.tracker = createTrackerByName("KCF");
				object.tracker->init(*mat, cv::Rect(s_areaSelector.m_selectedArea));
				object.numLost = 0;
				//object.rectFirst = s_selectedArea;
				s_objectList.push_back(object);
			}

			if (landmark.handflag >= 0.99) {
				s_isPalmByLmValid = true;
			} else {
				s_isPalmByLmValid = false;
			}
		} else {
			s_isPalmByLmValid = false;
		}
	}


	/* Track and display tracked objects */
	static int animCount = 0;;
	animCount++;
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
			drawRing(*mat, rect, cv::Scalar(255, 255, 205), animCount);
			drawText(*mat, rect, cv::Scalar(207, 161, 69), "Target", animCount);
			it->numLost = 0;
			it++;
		} else {
			PRINT("lost\n");
			if (++(it->numLost) > 100) {
				PRINT("delete\n");
				it = s_objectList.erase(it);
				tracker.release();
			} else {
				it++;
			}
		}
	}

	return 0;
}

