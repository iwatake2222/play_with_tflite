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

void AreaSelector::run(HandLandmark::HAND_LANDMARK &handLandmark)
{
	if (handLandmark.handflag > 0.9) {
		m_cntHandIsUntrusted = 0;	 //clear counter
		int fingerStatus = checkIfClosed(handLandmark);
		if (fingerStatus == -1) m_status = STATUS_AREA_SELECT_INIT;
		switch (m_status) {
		case STATUS_AREA_SELECT_INIT:
			if (fingerStatus == 1) {	/* Open -> Closed */
				m_status = STATUS_AREA_SELECT_START;
				m_startPoint.x = (int)(handLandmark.pos[4].x + handLandmark.pos[8].x) / 2;
				m_startPoint.y = (int)(handLandmark.pos[4].y + handLandmark.pos[8].y) / 2;
			}
			break;
		case STATUS_AREA_SELECT_START:
			/* update start position */
			m_startPoint.x = (int)(handLandmark.pos[4].x + handLandmark.pos[8].x) / 2;
			m_startPoint.y = (int)(handLandmark.pos[4].y + handLandmark.pos[8].y) / 2;
			if (fingerStatus == 1) {
				m_status = STATUS_AREA_SELECT_START;
			} else if (fingerStatus == 0) {
				m_status = STATUS_AREA_SELECT_DRAG;
			}
			break;
		case STATUS_AREA_SELECT_DRAG:
			m_selectedArea.x = std::min(m_startPoint.x, (int)((handLandmark.pos[4].x + handLandmark.pos[8].x) / 2));
			m_selectedArea.y = std::min(m_startPoint.y, (int)((handLandmark.pos[4].y + handLandmark.pos[8].y) / 2));
			m_selectedArea.width = (int)std::abs(m_startPoint.x - (handLandmark.pos[4].x + handLandmark.pos[8].x) / 2);
			m_selectedArea.height = (int)std::abs(m_startPoint.y - (handLandmark.pos[4].y + handLandmark.pos[8].y) / 2);
			if (fingerStatus == 1) {
				m_status = STATUS_AREA_SELECT_SELECTED;
			} else if (fingerStatus == 0) {
				m_status = STATUS_AREA_SELECT_DRAG;
			}
			break;
		case STATUS_AREA_SELECT_SELECTED:
			m_status = STATUS_AREA_SELECT_END;
			break;
		case STATUS_AREA_SELECT_END:
			if (fingerStatus == 0) {	/* Closed -> Open */
				m_status = STATUS_AREA_SELECT_INIT;
			}
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

