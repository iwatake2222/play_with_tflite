
#ifndef AREA_SELECTOR_
#define AREA_SELECTOR_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "hand_landmark_engine.h"


class AreaSelector
{
public:
	typedef enum {
		STATUS_AREA_SELECT_INIT,
		STATUS_AREA_SELECT_DRAG,
		STATUS_AREA_SELECT_SELECTED,
	} STATUS;

public:
	AreaSelector();
	~AreaSelector();
	void run(const HandLandmarkEngine::HAND_LANDMARK &hand_landmark);

private:
	// 0: open, 1: closed, -1 invalid
	//int32_t checkIfClosed(HandLandmark::HAND_LANDMARK &hand_landmark);

	// 0: index and middle, 1: index only, -1 other
	int32_t checkIfPointing(const HandLandmarkEngine::HAND_LANDMARK &hand_landmark);
	int32_t removeChattering(int32_t fingerStatus);

public:
	STATUS m_status;
	cv::Point m_startPoint;
	cv::Rect m_selectedArea;
	int32_t m_cntHandIsUntrusted;
	int32_t m_fingerStatus;
	int32_t m_cntToRemoveChattering;

};

#endif
