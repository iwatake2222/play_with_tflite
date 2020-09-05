
#ifndef INFERENCE_HELPER_TENSORFLOW_LITE_
#define INFERENCE_HELPER_TENSORFLOW_LITE_

#include "InferenceHelper.h"

/* for Tensorflow Lite */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

class InferenceHelperTensorflowLite : public InferenceHelper {
public:
	InferenceHelperTensorflowLite();
	~InferenceHelperTensorflowLite() override {};
	int initialize(const char *modelFilename, int numThreads) override;
	int initialize(const char *modelFilename, const int numThreads, std::vector<std::pair<const char*, const void*>> customOps) override;
	int finalize(void) override;
	int invoke(void) override;
	int getTensorByName(const char *name, TensorInfo *tensorInfo) override;
	int getTensorByIndex(const int index, TensorInfo *tensorInfo) override;
	int setBufferToTensorByName(const char *name, void *data, const int dataSize) override;
	int setBufferToTensorByIndex(const int index, void *data, const int dataSize) override;

private:
	int getIndexByName(const char *name);
	void displayModelInfo(const tflite::Interpreter* interpreter);
	TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src);

private:
	std::unique_ptr<tflite::FlatBufferModel> m_model;
	std::unique_ptr <tflite::ops::builtin::BuiltinOpResolver> m_resolver;
	std::unique_ptr<tflite::Interpreter> m_interpreter;
	TfLiteDelegate* m_delegate;
};

#endif
