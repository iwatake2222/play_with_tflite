/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
#include "edgetpu.h"
#include "edgetpu_c.h"
#endif

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

#include "InferenceHelperTensorflowLite.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[InferenceHelperTensorflowLite] " __VA_ARGS__)

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }


/*** Function ***/
InferenceHelperTensorflowLite::InferenceHelperTensorflowLite()
{
	m_resolver.reset(new tflite::ops::builtin::BuiltinOpResolver());
}

int InferenceHelperTensorflowLite::initialize(const char *modelFilename, int numThreads, std::vector<std::pair<const char*, const void*>> customOps)
{
	for (auto op : customOps) {
		m_resolver->AddCustom(op.first, (const TfLiteRegistration*)op.second);
	}

	return initialize(modelFilename, numThreads);
}

int InferenceHelperTensorflowLite::initialize(const char *modelFilename, int numThreads)
{
	/* Create interpreter */
	m_model = tflite::FlatBufferModel::BuildFromFile((std::string(modelFilename) + ".tflite").c_str());
	CHECK(m_model != nullptr);

	tflite::InterpreterBuilder builder(*m_model, *m_resolver);
	builder(&m_interpreter);
	CHECK(m_interpreter != nullptr);

	 m_interpreter->SetNumThreads(numThreads);

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
	if (m_helperType == TENSORFLOW_LITE_EDGETPU) {
		size_t num_devices;
		std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
		CHECK(num_devices > 0);
		const auto& device = devices.get()[0];
		m_delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
		m_interpreter->ModifyGraphWithDelegate(m_delegate);
	}
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
	if (m_helperType == TENSORFLOW_LITE_GPU) {
		auto options = TfLiteGpuDelegateOptionsV2Default();
		options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
		options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
		m_delegate = TfLiteGpuDelegateV2Create(&options);
		m_interpreter->ModifyGraphWithDelegate(m_delegate);
	}
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
	if (m_helperType == TENSORFLOW_LITE_XNNPACK) {
		auto options = TfLiteXNNPackDelegateOptionsDefault();
		options.num_threads = numThreads;
		m_delegate = TfLiteXNNPackDelegateCreate(&options);
		m_interpreter->ModifyGraphWithDelegate(m_delegate);
	}
#endif
	CHECK(m_interpreter->AllocateTensors() == kTfLiteOk);

	/* Get model information */
	displayModelInfo(m_interpreter.get());

	return 0;
}

int InferenceHelperTensorflowLite::finalize(void)
{
	m_model.reset();
	m_resolver.reset();
	m_interpreter.reset();

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
	if (m_helperType == TENSORFLOW_LITE_EDGETPU) {
		edgetpu_free_delegate(m_delegate);
	}
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
	if (m_helperType == TENSORFLOW_LITE_GPU) {
		TfLiteGpuDelegateV2Delete(m_delegate);
	}
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
	if (m_helperType == TENSORFLOW_LITE_XNNPACK) {
		TfLiteXNNPackDelegateDelete(m_delegate);
	}
#endif
	return 0;
}

int InferenceHelperTensorflowLite::invoke(void)
{
	CHECK(m_interpreter->Invoke() == kTfLiteOk)
	return 0;
}


int InferenceHelperTensorflowLite::getTensorByName(const char *name, TensorInfo *tensorInfo)
{
	int index = getIndexByName(name);
	if (index == -1) {
		PRINT("invalid name: %s\n", name);
		return -1;
	}

	return getTensorByIndex(index, tensorInfo);
}

int InferenceHelperTensorflowLite::getTensorByIndex(const int index, TensorInfo *tensorInfo)
{
	const TfLiteTensor* tensor = m_interpreter->tensor(index);

	tensorInfo->index = index;

	for (int i = 0; i < tensor->dims->size; i++) {
		tensorInfo->dims.push_back(tensor->dims->data[i]);
	}


	switch (tensor->type) {
	case kTfLiteUInt8:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_UINT8;
		tensorInfo->data = m_interpreter->typed_tensor<uint8_t>(index);
		tensorInfo->quant.scale = tensor->params.scale;
		tensorInfo->quant.zeroPoint = tensor->params.zero_point;
		break;
	case kTfLiteFloat32:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_FP32;
		tensorInfo->data = m_interpreter->typed_tensor<float>(index);
			break;
	case kTfLiteInt32:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_INT32;
		tensorInfo->data = m_interpreter->typed_tensor<int32_t>(index);
		break;
	case kTfLiteInt64:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_INT64;
		tensorInfo->data = m_interpreter->typed_tensor<int64_t>(index);
		break;
	default:
		CHECK(false);
	}
	return 0;
}

int InferenceHelperTensorflowLite::setBufferToTensorByName(const char *name, void *data, const int dataSize)
{
	int index = getIndexByName(name);
	if (index == -1) {
		PRINT("invalid name: %s\n", name);
		return -1;
	}

	return setBufferToTensorByIndex(index, data, dataSize);
}

int InferenceHelperTensorflowLite::setBufferToTensorByIndex(const int index, void *data, const int dataSize)
{
	const TfLiteTensor* tensor = m_interpreter->tensor(index);
	const int modelInputHeight = tensor->dims->data[1];
	const int modelInputWidth = tensor->dims->data[2];
	const int modelInputChannel = tensor->dims->data[3];

	if (tensor->type == kTfLiteUInt8) {
		CHECK(sizeof(int8_t) * 1 * modelInputHeight * modelInputWidth * modelInputChannel == (size_t)dataSize);
		/* Need deep copy quantization parameters */
		/* reference: https://github.com/google-coral/edgetpu/blob/master/src/cpp/basic/basic_engine_native.cc */
		/* todo: release them */
		const TfLiteAffineQuantization* inputQuantParams = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
		TfLiteQuantization inputQuantClone;
		inputQuantClone = tensor->quantization;
		TfLiteAffineQuantization* inputQuantParamsClone = reinterpret_cast<TfLiteAffineQuantization*>(malloc(sizeof(TfLiteAffineQuantization)));
		inputQuantParamsClone->scale = TfLiteFloatArrayCopy(inputQuantParams->scale);
		inputQuantParamsClone->zero_point = TfLiteIntArrayCopy(inputQuantParams->zero_point);
		inputQuantParamsClone->quantized_dimension = inputQuantParams->quantized_dimension;
		inputQuantClone.params = inputQuantParamsClone;

		//memcpy(m_interpreter->data.int8, data, sizeof(int8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
		m_interpreter->SetTensorParametersReadOnly(
			index, tensor->type, tensor->name,
			std::vector<int>(tensor->dims->data, tensor->dims->data + tensor->dims->size),
			inputQuantClone,	// use copied parameters
			(const char*)data, dataSize);
	} else {
		CHECK(sizeof(float) * 1 * modelInputHeight * modelInputWidth * modelInputChannel == (size_t)dataSize);
		//memcpy(m_interpreter->data.f, data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
		m_interpreter->SetTensorParametersReadOnly(
			index, tensor->type, tensor->name,
			std::vector<int>(tensor->dims->data, tensor->dims->data + tensor->dims->size),
			tensor->quantization,
			(const char*)data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}
	return 0;
}

TfLiteFloatArray* InferenceHelperTensorflowLite::TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
	if (!src) return nullptr;
	TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(
		malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
	if (!ret) return nullptr;
	ret->size = src->size;
	std::memcpy(ret->data, src->data, src->size * sizeof(float));
	return ret;
}


int InferenceHelperTensorflowLite::getIndexByName(const char *name)
{
	int index = -1;
	for (auto i : m_interpreter->inputs()) {
		const TfLiteTensor* tensor = m_interpreter->tensor(i);
		if (strcmp(tensor->name, name) == 0) {
			index = i;
			break;
		}
	}
	if (index == -1) {
		for (auto i : m_interpreter->outputs()) {
			const TfLiteTensor* tensor = m_interpreter->tensor(i);
			if (strcmp(tensor->name, name) == 0) {
				index = i;
				break;
			}
		}
	}
	return index;
}

void InferenceHelperTensorflowLite::displayModelInfo(const tflite::Interpreter* interpreter)
{
	const auto& inputIndices = interpreter->inputs();
	int inputNum = (int)inputIndices.size();
	PRINT("Input num = %d\n", inputNum);
	for (int i = 0; i < inputNum; i++) {
		auto* tensor = interpreter->tensor(inputIndices[i]);
		PRINT("    tensor[%d]->name: %s\n", i, tensor->name);
		for (int j = 0; j < tensor->dims->size; j++) {
			PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			PRINT("    tensor[%d]->type: quantized\n", i);
			PRINT("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			PRINT("    tensor[%d]->type: not quantized\n", i);
		}
	}

	const auto& outputIndices = interpreter->outputs();
	int outputNum = (int)outputIndices.size();
	PRINT("Output num = %d\n", outputNum);
	for (int i = 0; i < outputNum; i++) {
		auto* tensor = interpreter->tensor(outputIndices[i]);
		PRINT("    tensor[%d]->name: %s\n", i, tensor->name);
		for (int j = 0; j < tensor->dims->size; j++) {
			PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			PRINT("    tensor[%d]->type: quantized\n", i);
			PRINT("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			PRINT("    tensor[%d]->type: not quantized\n", i);
		}
	}
}
