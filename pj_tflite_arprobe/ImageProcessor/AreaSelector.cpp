/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "AreaSelector.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[AreaSelector] " __VA_ARGS__)

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

const int INDEX_THUMB_FINGER_START = 1;
const int INDEX_THUMB_FINGER_END = 4;
const int INDEX_INDEX_FINGER_START = 5;
const int INDEX_INDEX_FINGER_END = 8;
const int INDEX_MIDDLE_FINGER_START = 9;
const int INDEX_MIDDLE_FINGER_END = 12;
const int INDEX_RING_FINGER_START = 13;
const int INDEX_RING_FINGER_END = 16;
const int INDEX_LITTLE_FINGER_START = 17;
const int INDEX_LITTLE_FINGER_END = 20;

/*** Function ***/
AreaSelector::AreaSelector()
{
	m_status = STATUS_AREA_SELECT_INIT;
	m_cntHandIsUntrusted = 0;
	m_fingerStatus = -1;
	m_cntToRemoveChattering = 0;
}

AreaSelector::~AreaSelector()
{
}

void AreaSelector::run(HandLandmark::HAND_LANDMARK &handLandmark)
{
	int fingerStatus = -1;
	if (handLandmark.handflag > 0.9) {
		fingerStatus = checkIfPointing(handLandmark);
		PRINT("fingerStatus = %d\n", fingerStatus);
		fingerStatus = removeChattering(fingerStatus);
		PRINT("fingerStatus = %d\n", fingerStatus);
	}
	
	if (fingerStatus == -1) {
		m_cntHandIsUntrusted++;
		if (m_cntHandIsUntrusted > 10) {
			m_status = STATUS_AREA_SELECT_INIT;
			//m_cntHandIsUntrusted = 0;
			//m_cntToRemoveChattering = 0;
		}
	} else {
		m_cntHandIsUntrusted = 0;	 //clear counter
	}
	
	switch (m_status) {
	case STATUS_AREA_SELECT_INIT:
		m_selectedArea.x = 0;
		m_selectedArea.y = 0;
		m_selectedArea.width = 0;
		m_selectedArea.height = 0;
		if (fingerStatus == 1) {
			m_status = STATUS_AREA_SELECT_DRAG;
			m_startPoint.x = (int)handLandmark.pos[INDEX_INDEX_FINGER_END].x;
			m_startPoint.y = (int)handLandmark.pos[INDEX_INDEX_FINGER_END].y;
		}
		break;
	case STATUS_AREA_SELECT_DRAG:
		if (fingerStatus != -1) {
			m_selectedArea.x = std::min(m_startPoint.x, (int)(handLandmark.pos[INDEX_INDEX_FINGER_END].x));
			m_selectedArea.y = std::min(m_startPoint.y, (int)(handLandmark.pos[INDEX_INDEX_FINGER_END].y));
			m_selectedArea.width = (int)std::abs(m_startPoint.x - handLandmark.pos[INDEX_INDEX_FINGER_END].x);
			m_selectedArea.height = (int)std::abs(m_startPoint.y - handLandmark.pos[INDEX_INDEX_FINGER_END].y);
			if (fingerStatus == 0) {
				m_status = STATUS_AREA_SELECT_SELECTED;
			} else if (fingerStatus == 1) {
				m_status = STATUS_AREA_SELECT_DRAG;
			}
		}
		break;
	case STATUS_AREA_SELECT_SELECTED:
		m_status = STATUS_AREA_SELECT_INIT;
		break;
	default:
		break;
	}
}

int AreaSelector::removeChattering(int value)
{
	if (m_fingerStatus != value) {
		m_cntToRemoveChattering++;
		if (m_cntToRemoveChattering > 10) {
			m_cntToRemoveChattering = 0;
			m_fingerStatus = value;
		}
	} else {
		m_cntToRemoveChattering = 0;
	}

	if (value == -1) {
		return -1;
	} else {
		return m_fingerStatus;
	}
}

// 1: pointed by index finger, 1: pointed by index and middle fingers, -1 invalid
int AreaSelector::checkIfPointing(HandLandmark::HAND_LANDMARK &handLandmark)
{
	const int POINTED_INDEX = 0;
	const int POINTED_INDEX_MIDDLE = 1;
	const int INVALID = -1;

	const double MAX_GRADIENT = 30;
	const double THRESH_GRADIENT_INDEX_FINGER = 0.6;
	double threshGradient = (m_status == STATUS_AREA_SELECT_INIT) ? 0.6 : 0.8;
	double threshDistance = (m_status == STATUS_AREA_SELECT_INIT) ? 0.3 : 0.6;

	/* calculate gradient of each finger */
	double gradientIndexFinger = MAX_GRADIENT;
	if ((handLandmark.pos[INDEX_INDEX_FINGER_END].x - handLandmark.pos[INDEX_INDEX_FINGER_START].x) != 0) {
		gradientIndexFinger = (double)(handLandmark.pos[INDEX_INDEX_FINGER_END].y - handLandmark.pos[INDEX_INDEX_FINGER_START].y) / (handLandmark.pos[INDEX_INDEX_FINGER_END].x - handLandmark.pos[INDEX_INDEX_FINGER_START].x);
		if (gradientIndexFinger > MAX_GRADIENT) gradientIndexFinger = MAX_GRADIENT;
	}
	double gradientMiddleFinger = MAX_GRADIENT;
	if ((handLandmark.pos[INDEX_MIDDLE_FINGER_END].x - handLandmark.pos[INDEX_MIDDLE_FINGER_START].x) != 0) {
		gradientMiddleFinger = (double)(handLandmark.pos[INDEX_MIDDLE_FINGER_END].y - handLandmark.pos[INDEX_MIDDLE_FINGER_START].y) / (handLandmark.pos[INDEX_MIDDLE_FINGER_END].x - handLandmark.pos[INDEX_MIDDLE_FINGER_START].x);
		if (gradientMiddleFinger > MAX_GRADIENT) gradientMiddleFinger = MAX_GRADIENT;
	}
	double gradientRingFinger = MAX_GRADIENT;
	if ((handLandmark.pos[INDEX_RING_FINGER_END].x - handLandmark.pos[INDEX_RING_FINGER_START].x) != 0) {
		gradientRingFinger = (double)(handLandmark.pos[INDEX_RING_FINGER_END].y - handLandmark.pos[INDEX_RING_FINGER_START].y) / (handLandmark.pos[INDEX_RING_FINGER_END].x - handLandmark.pos[INDEX_RING_FINGER_START].x);
		if (gradientRingFinger > MAX_GRADIENT) gradientRingFinger = MAX_GRADIENT;
	}
	double gradientLittleFinger = MAX_GRADIENT;
	if ((handLandmark.pos[INDEX_LITTLE_FINGER_END].x - handLandmark.pos[INDEX_LITTLE_FINGER_START].x) != 0) {
		gradientLittleFinger = (double)(handLandmark.pos[INDEX_LITTLE_FINGER_END].y - handLandmark.pos[INDEX_LITTLE_FINGER_START].y) / (handLandmark.pos[INDEX_LITTLE_FINGER_END].x - handLandmark.pos[INDEX_LITTLE_FINGER_START].x);
		if (gradientLittleFinger > MAX_GRADIENT) gradientLittleFinger = MAX_GRADIENT;
	}
	PRINT("index = %5.3lf, middle = %5.3lf, ring = %5.3lf, little = %5.3lf\n", gradientIndexFinger, gradientMiddleFinger, gradientRingFinger, gradientLittleFinger);

	/* calculate gradient of each joint of each finger */
	std::vector<double> gradientIndexFingers;
	for (int i = INDEX_INDEX_FINGER_START; i < INDEX_INDEX_FINGER_END; i++) {
		double dx = handLandmark.pos[i + 1].x - handLandmark.pos[i].x;
		double dy = handLandmark.pos[i + 1].y - handLandmark.pos[i].y;
		double gradient = MAX_GRADIENT;
		if (dx != 0) gradient = std::min(dy / dx, MAX_GRADIENT);
		gradientIndexFingers.push_back(gradient);
		PRINT("index = %5.3lf\n", gradient);
	}

	std::vector<double> gradientMiddleFingers;
	for (int i = INDEX_MIDDLE_FINGER_START; i < INDEX_MIDDLE_FINGER_END; i++) {
		double dx = handLandmark.pos[i + 1].x - handLandmark.pos[i].x;
		double dy = handLandmark.pos[i + 1].y - handLandmark.pos[i].y;
		double gradient = MAX_GRADIENT;
		if (dx != 0) gradient = std::min(dy / dx, MAX_GRADIENT);
		gradientMiddleFingers.push_back(gradient);
		PRINT("middle = %5.3lf\n", gradient);
	}

	std::vector<double> gradientRingFingers;
	for (int i = INDEX_RING_FINGER_START; i < INDEX_RING_FINGER_END; i++) {
		double dx = handLandmark.pos[i + 1].x - handLandmark.pos[i].x;
		double dy = handLandmark.pos[i + 1].y - handLandmark.pos[i].y;
		double gradient = MAX_GRADIENT;
		if (dx != 0) gradient = std::min(dy / dx, MAX_GRADIENT);
		gradientRingFingers.push_back(gradient);
		PRINT("ring = %5.3lf\n", gradient);
	}
	std::vector<double> gradientLittleFingers;
	for (int i = INDEX_LITTLE_FINGER_START; i < INDEX_LITTLE_FINGER_END; i++) {
		double dx = handLandmark.pos[i + 1].x - handLandmark.pos[i].x;
		double dy = handLandmark.pos[i + 1].y - handLandmark.pos[i].y;
		double gradient = MAX_GRADIENT;
		if (dx != 0) gradient = std::min(dy / dx, MAX_GRADIENT);
		gradientLittleFingers.push_back(gradient);
		PRINT("little = %5.3lf\n", gradient);
	}

	/* check if the ring and little fingers are held */
	bool fingerIsHeld = true;
	if (handLandmark.pos[INDEX_INDEX_FINGER_END].y < handLandmark.pos[INDEX_INDEX_FINGER_START].y) {
		for (int i = INDEX_RING_FINGER_START; i < INDEX_RING_FINGER_END; i++) {
			if (handLandmark.pos[INDEX_RING_FINGER_END].y < handLandmark.pos[i].y) fingerIsHeld = false;
		}
		for (int i = INDEX_LITTLE_FINGER_START; i < INDEX_LITTLE_FINGER_END; i++) {
			if (handLandmark.pos[INDEX_LITTLE_FINGER_END].y < handLandmark.pos[i].y) fingerIsHeld = false;
		}
	} else {
		for (int i = INDEX_RING_FINGER_START; i < INDEX_RING_FINGER_END; i++) {
			if (handLandmark.pos[INDEX_RING_FINGER_END].y > handLandmark.pos[i].y) fingerIsHeld = false;
		}
		for (int i = INDEX_LITTLE_FINGER_START; i < INDEX_LITTLE_FINGER_END; i++) {
			if (handLandmark.pos[INDEX_LITTLE_FINGER_END].y > handLandmark.pos[i].y) fingerIsHeld = false;
		}
	}
	PRINT("fingerIsHeld = %d\n", fingerIsHeld);
	if (fingerIsHeld == false) return INVALID;

	/* check if the index finger is straight */
	for (int i = 0; i < gradientIndexFingers.size() - 2; i++) {	// ignore the last node (the last node tends not to be straight)
		if (std::abs((gradientIndexFingers[i] - gradientIndexFingers[i + 1]) / gradientIndexFingers[i]) > THRESH_GRADIENT_INDEX_FINGER) return INVALID;
		if (gradientIndexFingers[i] * gradientIndexFingers[i + 1] < 0) return INVALID;
	}

	/* check if the middle finger is straight */
	for (int i = 0; i < gradientMiddleFingers.size() - 2; i++) {
		if (std::abs((gradientMiddleFingers[i] - gradientMiddleFingers[i + 1]) / gradientMiddleFingers[i]) > threshGradient) return POINTED_INDEX;
		if (gradientMiddleFingers[i] * gradientMiddleFingers[i + 1] < 0)  return POINTED_INDEX;
	}

	/*  check if teh middle finger has the same gradient as index finger */
	if (std::abs((gradientIndexFinger - gradientMiddleFinger) / gradientIndexFinger) > threshGradient) return POINTED_INDEX;
	if (gradientIndexFinger * gradientMiddleFinger < 0)  return POINTED_INDEX;

	/*  chec the distance b/w index and middle finger */
	double distance = pow(handLandmark.pos[INDEX_INDEX_FINGER_END].x - handLandmark.pos[INDEX_MIDDLE_FINGER_END].x, 2) + pow(handLandmark.pos[INDEX_INDEX_FINGER_END].y - handLandmark.pos[INDEX_MIDDLE_FINGER_END].y, 2);
	distance = sqrt(distance);
	double baseDistance = pow(handLandmark.pos[INDEX_INDEX_FINGER_START].x - handLandmark.pos[INDEX_INDEX_FINGER_END].x, 2) + pow(handLandmark.pos[INDEX_INDEX_FINGER_START].y - handLandmark.pos[INDEX_INDEX_FINGER_END].y, 2);
	baseDistance = sqrt(baseDistance);
	if (distance > baseDistance * threshDistance) return POINTED_INDEX;

	return POINTED_INDEX_MIDDLE;
}

//int AreaSelector::checkIfClosed(HandLandmark::HAND_LANDMARK &handLandmark)
//{
//	const int OPEN = 0;
//	const int CLOSE = 1;
//	const int INVALID = -1;
//	static int s_status = OPEN;
//	double thresholdOpen2Close = 0.2;
//	double thresholdClose2Open = 0.4;
//
//	//PRINT("%f %f\n", landmark.pos[indexThumbFinger].z, landmark.pos[indexIndexFinger].z);
//	///* check z pos (z pos should be plus) */
//	//if (landmark.pos[indexThumbFinger].z > 0 || landmark.pos[indexIndexFinger].z > 0) {
//	//	PRINT("invalid 1\n");
//	//	return INVALID;
//	//}
//
//	/* check y pos (Thumb must be lower than index finger) */
//	//for (int i = 1; i <= indexThumbFinger; i++) {
//	//	for (int j = indexThumbFinger + 1; j <= indexIndexFinger; j++) {
//	//		if (landmark.pos[i].y < landmark.pos[j].y) {
//	//			PRINT("invalid2 \n");
//	//			return INVALID;
//	//		}
//	//	}
//	//}
//
//	/* check shape (shape must be horizontally long) */
//	int thumbW = (int)std::abs(handLandmark.pos[1].x - handLandmark.pos[4].x);
//	int thumbH = (int)std::abs(handLandmark.pos[1].y - handLandmark.pos[4].y);
//	double thumbLong = std::sqrt(thumbW * thumbW + thumbH * thumbH);
//	if ((double)thumbH / thumbW > 0.7) {
//		PRINT("invalid 3\n");
//		return INVALID;
//	}
//
//	/* check if close */
//	int distW = (int)(handLandmark.pos[indexThumbFinger].x - handLandmark.pos[indexIndexFinger].x);
//	int distH = (int)(handLandmark.pos[indexThumbFinger].y - handLandmark.pos[indexIndexFinger].y);
//	double dist = std::sqrt(distW * distW + distH * distH);
//
//	double threshold = (s_status == CLOSE) ? thresholdClose2Open : thresholdOpen2Close;
//	if (dist < thumbLong * threshold) {
//		PRINT("close\n");
//		s_status = CLOSE;
//	} else {
//		PRINT("open\n");
//		s_status = OPEN;
//	}
//	return s_status;
//}
//
