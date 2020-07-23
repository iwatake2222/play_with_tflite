/*** Include ***/
/* for general */
#include <stdint.h>
#include <stdio.h>
#include <fstream> 
#include <vector>
#include <string>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for Tensorflow Lite */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

/* for Edge TPU */
#include "edgetpu.h"
#include "model_utils.h"

/*** Macro ***/
/* Model parameters */
#define MODEL_FILENAME RESOURCE_DIR"/model/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
#define LABEL_NAME     RESOURCE_DIR"/model/imagenet_labels.txt"
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL 3

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 1000

/*** Function ***/
static void readLabel(const char* filename, std::vector<std::string> &labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		printf("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while(getline(ifs, str)) {
		labels.push_back(str);
	}
}

int main()
{
	/*** Initialize ***/
	/* read label */
	std::vector<std::string> labels;
	readLabel(LABEL_NAME, labels);

	/* Create interpreter */
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
	std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
	std::unique_ptr<tflite::Interpreter> interpreter = coral::BuildEdgeTpuInterpreter(*model, edgetpu_context.get());

	/*** Process for each frame ***/
	/* Read input image data */
	cv::Mat inputImage = cv::imread(RESOURCE_DIR"/parrot.jpg");
	/* Pre-process */
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	cv::resize(inputImage, inputImage, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
	std::vector<uint8_t> inputData(inputImage.data, inputImage.data + (inputImage.cols * inputImage.rows * inputImage.elemSize()));

	/* Run inference */
	const auto& scores = coral::RunInference(inputData, interpreter.get());

	/* Retrieve the result */
	int maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
	float maxScore = *std::max_element(scores.begin(), scores.end());
	printf("%s (%.3f)\n", labels[maxIndex].c_str(), maxScore);


	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		coral::RunInference(inputData, interpreter.get());
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
	
	return 0;
}

