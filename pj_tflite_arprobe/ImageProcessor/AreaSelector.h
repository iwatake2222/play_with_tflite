
#ifndef AREA_SELECTOR_
#define AREA_SELECTOR_

/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "HandLandmark.h"

class AreaSelector
{
public:
	typedef enum {
		STATUS_AREA_SELECT_INIT,
		STATUS_AREA_SELECT_START,
		STATUS_AREA_SELECT_DRAG,
		STATUS_AREA_SELECT_SELECTED,
		STATUS_AREA_SELECT_END,
	} STATUS;

public:
	AreaSelector();
	~AreaSelector();
	void run(HandLandmark::HAND_LANDMARK &handLandmark);

private:
	// 0: open, 1: closed, -1 invalid
	int checkIfClosed(HandLandmark::HAND_LANDMARK &handLandmark);

public:
	STATUS m_status;
	cv::Point m_startPoint;
	cv::Rect m_selectedArea;

};

#endif
