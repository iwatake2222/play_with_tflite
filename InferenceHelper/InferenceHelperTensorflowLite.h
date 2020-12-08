#ifndef INFERENCE_HELPER_TENSORFLOW_LITE_
#define INFERENCE_HELPER_TENSORFLOW_LITE_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for Tensorflow Lite */
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>

/* for My modules */
#include "InferenceHelper.h"

class InferenceHelperTensorflowLite : public InferenceHelper {
public:
	InferenceHelperTensorflowLite();
	~InferenceHelperTensorflowLite() override;
	int32_t setNumThread(const int32_t numThread) override;
	int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
	int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
	int32_t finalize(void) override;
	int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
	int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
	int32_t getInputTensorInfo(InputTensorInfo& tensorInfo);
	int32_t getOutputTensorInfo(OutputTensorInfo& tensorInfo);
	void convertNormalizeParameters(InputTensorInfo& tensorInfo);
	void displayModelInfo(const tflite::Interpreter& interpreter);

	int32_t setBufferToTensor(int32_t index, void *data);


private:
	std::unique_ptr<tflite::FlatBufferModel> m_model;
	std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> m_resolver;
	std::unique_ptr<tflite::Interpreter> m_interpreter;
	TfLiteDelegate* m_delegate;

	int32_t m_numThread;
};

#endif
