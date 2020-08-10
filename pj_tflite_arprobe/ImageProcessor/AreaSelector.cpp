/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "AreaSelector.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, "[AreaSelector] " __VA_ARGS__)
#else
#define PRINT(fmt, ...) printf("[AreaSelector] " fmt, __VA_ARGS__)
#endif

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/*** Function ***/
AreaSelector::AreaSelector()
{
	m_status = STATUS_AREA_SELECT_INIT;
	m_cntHandIsUntrusted = 0;
}

AreaSelector::~AreaSelector()
{
}

static int chattering(int value)
{
	static int s_finalValue = -1;
	static int cnt = 0;

	if (value == -1) return s_finalValue;

	if (s_finalValue != value) {
		cnt++;
		if (cnt > 10) {
			cnt = 0;
			s_finalValue = value;
		}
	} else {
		//cnt = 0;
	}

	return s_finalValue;
}

void AreaSelector::run(HandLandmark::HAND_LANDMARK &handLandmark)
{
	int fingerStatus = -1;
	if (handLandmark.handflag > 0.9) {
		fingerStatus = checkIfPointing(handLandmark);
		fingerStatus = chattering(fingerStatus);
		PRINT("fingerStatus = %d\n", fingerStatus);
	}
	
	if (fingerStatus != -1) {
		m_cntHandIsUntrusted = 0;	 //clear counter
		switch (m_status) {
		case STATUS_AREA_SELECT_INIT:
			if (fingerStatus == 1) {	/* Open -> Closed */
				m_status = STATUS_AREA_SELECT_DRAG;
				m_startPoint.x = (int)handLandmark.pos[8].x;
				m_startPoint.y = (int)handLandmark.pos[8].y;
			}
			break;
		case STATUS_AREA_SELECT_DRAG:
			m_selectedArea.x = std::min(m_startPoint.x, (int)(handLandmark.pos[8].x));
			m_selectedArea.y = std::min(m_startPoint.y, (int)(handLandmark.pos[8].y));
			m_selectedArea.width = (int)std::abs(m_startPoint.x - handLandmark.pos[8].x);
			m_selectedArea.height = (int)std::abs(m_startPoint.y - handLandmark.pos[8].y);
			if (fingerStatus == 0) {
				m_status = STATUS_AREA_SELECT_SELECTED;
			} else if (fingerStatus == 1) {
				m_status = STATUS_AREA_SELECT_DRAG;
			}
			break;
		case STATUS_AREA_SELECT_SELECTED:
			m_status = STATUS_AREA_SELECT_INIT;
			break;
		default:
			break;
		}
	} else {
		m_cntHandIsUntrusted++;
		if (m_cntHandIsUntrusted > 20) {
			m_status = STATUS_AREA_SELECT_INIT;
		}
	}
}

int AreaSelector::checkIfPointing(HandLandmark::HAND_LANDMARK &handLandmark)
{
	const int POINTED_INDEX = 0;
	const int POINTED_INDEX_MIDDLE = 1;
	const int INVALID = -1;
	const int indexIndexFingerStart = 5;
	const int indexIndexFingerEnd = 8;
	const int indexMiddleFingerStart = 9;
	const int indexMiddleFingerEnd = 12;

	std::vector<double> gradientIndexFingers;
	for (int i = indexIndexFingerStart; i < indexIndexFingerEnd; i++) {
		double dx = handLandmark.pos[i + 1].x - handLandmark.pos[i].x;
		double dy = handLandmark.pos[i + 1].y - handLandmark.pos[i].y;
		double gradient = 99;
		if (dx != 0) gradient = dy / dx;
		gradientIndexFingers.push_back(gradient);
		PRINT("index = %5.3lf\n", gradient);
	}

	std::vector<double> gradientMiddleFingers;
	for (int i = indexMiddleFingerStart; i < indexMiddleFingerEnd; i++) {
		double dx = handLandmark.pos[i + 1].x - handLandmark.pos[i].x;
		double dy = handLandmark.pos[i + 1].y - handLandmark.pos[i].y;
		double gradient = 99;
		if (dx != 0) gradient = dy / dx;
		gradientMiddleFingers.push_back(gradient);
		PRINT("middle = %5.3lf\n", gradient);
	}

	/* check if the index finger is straight */
	for (int i = 0; i < gradientIndexFingers.size() - 1; i++) {
		if (std::abs((gradientIndexFingers[i] - gradientIndexFingers[i + 1]) / gradientIndexFingers[i]) > 0.5) return INVALID;
		if (gradientIndexFingers[i] * gradientIndexFingers[i + 1] < 0)  return INVALID;
	}

	/* check if the middle finger is straight */
	for (int i = 0; i < gradientIndexFingers.size() - 1; i++) {
		if (std::abs((gradientMiddleFingers[i] - gradientMiddleFingers[i + 1]) / gradientMiddleFingers[i]) > 0.5) return POINTED_INDEX;
		if (gradientMiddleFingers[i] * gradientMiddleFingers[i + 1] < 0)  return POINTED_INDEX;
	}

	/*  check if teh middle finger has the same gradient as index finger */
	double gradientIndexFinger = (double)(handLandmark.pos[indexIndexFingerEnd].y - handLandmark.pos[indexIndexFingerStart].y) / (handLandmark.pos[indexIndexFingerEnd].x - handLandmark.pos[indexIndexFingerStart].x);
	double gradientMiddleFinger = (double)(handLandmark.pos[indexMiddleFingerEnd].y - handLandmark.pos[indexMiddleFingerStart].y) / (handLandmark.pos[indexMiddleFingerEnd].x - handLandmark.pos[indexMiddleFingerStart].x);
	PRINT("index = %5.3lf,  middle = %5.3lf\n", gradientIndexFinger, gradientMiddleFinger);
	if (std::abs((gradientIndexFinger - gradientMiddleFinger) / gradientIndexFinger) > 0.5) return POINTED_INDEX;
	if (gradientIndexFinger * gradientMiddleFinger < 0)  return POINTED_INDEX;

	/*  chec the distance b/w index and middle finger */
	double distance = pow(handLandmark.pos[indexIndexFingerEnd].x - handLandmark.pos[indexMiddleFingerEnd].x, 2) + pow(handLandmark.pos[indexIndexFingerEnd].y - handLandmark.pos[indexMiddleFingerEnd].y, 2);
	distance = sqrt(distance);
	double baseDistance = pow(handLandmark.pos[indexIndexFingerStart].x - handLandmark.pos[indexIndexFingerEnd].x, 2) + pow(handLandmark.pos[indexIndexFingerStart].y - handLandmark.pos[indexIndexFingerEnd].y, 2);
	baseDistance = sqrt(baseDistance);
	if (distance > baseDistance * 0.25) return POINTED_INDEX;

	return POINTED_INDEX_MIDDLE;
}

int AreaSelector::checkIfClosed(HandLandmark::HAND_LANDMARK &handLandmark)
{
	const int OPEN = 0;
	const int CLOSE = 1;
	const int INVALID = -1;
	const int indexThumbFinger = 4;
	const int indexIndexFinger = 8;
	static int s_status = OPEN;
	double thresholdOpen2Close = 0.2;
	double thresholdClose2Open = 0.4;

	//PRINT("%f %f\n", landmark.pos[indexThumbFinger].z, landmark.pos[indexIndexFinger].z);
	///* check z pos (z pos should be plus) */
	//if (landmark.pos[indexThumbFinger].z > 0 || landmark.pos[indexIndexFinger].z > 0) {
	//	PRINT("invalid 1\n");
	//	return INVALID;
	//}

	/* check y pos (Thumb must be lower than index finger) */
	//for (int i = 1; i <= indexThumbFinger; i++) {
	//	for (int j = indexThumbFinger + 1; j <= indexIndexFinger; j++) {
	//		if (landmark.pos[i].y < landmark.pos[j].y) {
	//			PRINT("invalid2 \n");
	//			return INVALID;
	//		}
	//	}
	//}

	/* check shape (shape must be horizontally long) */
	int thumbW = (int)std::abs(handLandmark.pos[1].x - handLandmark.pos[4].x);
	int thumbH = (int)std::abs(handLandmark.pos[1].y - handLandmark.pos[4].y);
	double thumbLong = std::sqrt(thumbW * thumbW + thumbH * thumbH);
	if ((double)thumbH / thumbW > 0.7) {
		PRINT("invalid 3\n");
		return INVALID;
	}

	/* check if close */
	int distW = (int)(handLandmark.pos[indexThumbFinger].x - handLandmark.pos[indexIndexFinger].x);
	int distH = (int)(handLandmark.pos[indexThumbFinger].y - handLandmark.pos[indexIndexFinger].y);
	double dist = std::sqrt(distW * distW + distH * distH);

	double threshold = (s_status == CLOSE) ? thresholdClose2Open : thresholdOpen2Close;
	if (dist < thumbLong * threshold) {
		PRINT("close\n");
		s_status = CLOSE;
	} else {
		PRINT("open\n");
		s_status = OPEN;
	}
	return s_status;
}

