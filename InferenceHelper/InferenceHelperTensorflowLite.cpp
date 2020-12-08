/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

/* for Tensorflow Lite */
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

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelperTensorflowLite.h"

/*** Macro ***/
#define TAG "InferenceHelperTensorflowLite"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperTensorflowLite::InferenceHelperTensorflowLite()
{
	m_numThread = 1;
	m_resolver.reset(new tflite::ops::builtin::BuiltinOpResolver());
}

InferenceHelperTensorflowLite::~InferenceHelperTensorflowLite()
{
}

int32_t InferenceHelperTensorflowLite::setNumThread(const int32_t numThread)
{
	m_numThread = numThread;
	return RET_OK;
}

int32_t InferenceHelperTensorflowLite::setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
{
	for (auto op : customOps) {
		m_resolver->AddCustom(op.first, (const TfLiteRegistration*)op.second);
	}
	return RET_OK;
}

int32_t InferenceHelperTensorflowLite::initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	/*** Create network ***/
	m_model = tflite::FlatBufferModel::BuildFromFile(modelFilename.c_str());
	if (m_model == nullptr) {
		PRINT_E("Failed to build model (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}

	tflite::InterpreterBuilder builder(*m_model, *m_resolver);
	builder(&m_interpreter);
	if (m_interpreter == nullptr) {
		PRINT_E("Failed to build interpreter (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}

	m_interpreter->SetNumThreads(m_numThread);

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
	if (m_helperType == TENSORFLOW_LITE_EDGETPU) {
		size_t num_devices;
		std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
		if (num_devices > 0) {
			const auto& device = devices.get()[0];
			m_delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
			m_interpreter->ModifyGraphWithDelegate(m_delegate);
		} else {
			PRINT_E("[WARNING] Edge TPU is not found\n");
		}
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
		options.num_threads = m_numThread;
		m_delegate = TfLiteXNNPackDelegateCreate(&options);
		m_interpreter->ModifyGraphWithDelegate(m_delegate);
	}
#endif

	/* Memo: If you get error around here in Visual Studio, please make sure you don't use Debug */
	if (m_interpreter->AllocateTensors() != kTfLiteOk) {
		PRINT_E("Failed to allocate tensors (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}

	/* Get model information */
	displayModelInfo(*m_interpreter);

	/* Check if input tensor name exists anddims are the same as described in the model. In case dims is unfixed, resize tensor size. Get id and type */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		if (getInputTensorInfo(inputTensorInfo) != RET_OK) {
			PRINT_E("Invalid input tensor info (%s)\n", inputTensorInfo.name.c_str());
			return RET_ERR;
		}
	}
	
	/* Check if output tensor name exists and get info (id, ptr to data, dims, type) */
	for (auto& outputTensorInfo : outputTensorInfoList) {
		if (getOutputTensorInfo(outputTensorInfo) != RET_OK) {
			PRINT_E("Invalid output tensor info (%s)\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}
	}

	/* Convert normalize parameter to speed up */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		convertNormalizeParameters(inputTensorInfo);
	}

	return RET_OK;
};


int32_t InferenceHelperTensorflowLite::finalize(void)
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
	return RET_OK;
}

int32_t InferenceHelperTensorflowLite::preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
{
	if (m_interpreter == nullptr) {
		PRINT_E("Interpreter is not built yet\n");
		return RET_ERR;
	}

	for (const auto& inputTensorInfo : inputTensorInfoList) {
		TfLiteTensor* tensor = m_interpreter->tensor(inputTensorInfo.id);
		if (!tensor) {
			PRINT_E("Invalid input name (%s, %d)\n", inputTensorInfo.name.c_str(), inputTensorInfo.id);
			return RET_ERR;
		}
		if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_IMAGE) {
			if ((inputTensorInfo.imageInfo.width != inputTensorInfo.imageInfo.cropWidth) || (inputTensorInfo.imageInfo.height != inputTensorInfo.imageInfo.cropHeight)) {
				PRINT_E("Crop is not supported\n");
				return  RET_ERR;
			}
			if ((inputTensorInfo.imageInfo.cropWidth != inputTensorInfo.tensorDims.width) || (inputTensorInfo.imageInfo.cropHeight != inputTensorInfo.tensorDims.height)) {
				PRINT_E("Resize is not supported\n");
				return  RET_ERR;
			}
			if (inputTensorInfo.imageInfo.channel != inputTensorInfo.tensorDims.channel) {
				PRINT_E("Color conversion is not supported\n");
				return  RET_ERR;
			}

			/* Normalize image (NHWC to NHWC)*/
			uint8_t* src = static_cast<uint8_t*>(inputTensorInfo.data);
			if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) {
				uint8_t* dst = m_interpreter->typed_tensor<uint8_t>(inputTensorInfo.id);
				memcpy(dst, src, sizeof(uint8_t) * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel);
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
				float* dst = m_interpreter->typed_tensor<float>(inputTensorInfo.id);
#pragma omp parallel for num_threads(m_numThread)
				for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
					for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
#if 1
						dst[i * inputTensorInfo.tensorDims.channel + c] = (src[i * inputTensorInfo.tensorDims.channel + c] - inputTensorInfo.normalize.mean[c]) * inputTensorInfo.normalize.norm[c];
#else
						dst[i * inputTensorInfo.tensorDims.channel + c] = (src[i * inputTensorInfo.tensorDims.channel + c] / 255.0f - inputTensorInfo.normalize.mean[c]) / inputTensorInfo.normalize.norm[c];
#endif
					}
				}
			} else {
				PRINT_E("Unsupported tensorType (%d)\n", inputTensorInfo.tensorType);
				return RET_ERR;
			}

		} else if ( (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) || (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NCHW) ){
			if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) {
				uint8_t* dst = m_interpreter->typed_tensor<uint8_t>(inputTensorInfo.id);
				uint8_t* src = static_cast<uint8_t*>(inputTensorInfo.data);
				if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {
					//memcpy(dst, src, sizeof(uint8_t) * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel);
					setBufferToTensor(inputTensorInfo.id, src);
				} else {	/* NCHW -> NHWC */
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
						for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
							dst[i * inputTensorInfo.tensorDims.channel + c] = src[c * (inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height) + i];
						}
					}
				}
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
				float* dst = m_interpreter->typed_tensor<float>(inputTensorInfo.id);
				float* src = static_cast<float*>(inputTensorInfo.data);
				if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {	/* NHWC -> NHWC */
					//memcpy(dst, src, sizeof(float) * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel);
					setBufferToTensor(inputTensorInfo.id, src);
				} else {	/* NCHW -> NHWC */
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
						for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
							dst[i * inputTensorInfo.tensorDims.channel + c] = src[c * (inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height) + i];
						}
					}
				}
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_INT32) {
				int32_t* dst = m_interpreter->typed_tensor<int32_t>(inputTensorInfo.id);
				int32_t* src = static_cast<int32_t*>(inputTensorInfo.data);
				if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {	/* NHWC -> NHWC */
					//memcpy(dst, src, sizeof(int32_t) * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel);
					setBufferToTensor(inputTensorInfo.id, src);
				} else {	/* NCHW -> NHWC */
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
						for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
							dst[i * inputTensorInfo.tensorDims.channel + c] = src[c * (inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height) + i];
						}
					}
				}
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_INT64) {
				int64_t* dst = m_interpreter->typed_tensor<int64_t>(inputTensorInfo.id);
				int64_t* src = static_cast<int64_t*>(inputTensorInfo.data);
				if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {	/* NHWC -> NHWC */
					//memcpy(dst, src, sizeof(int64_t) * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel);
					setBufferToTensor(inputTensorInfo.id, src);
				} else {	/* NCHW -> NHWC */
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
						for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
							dst[i * inputTensorInfo.tensorDims.channel + c] = src[c * (inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height) + i];
						}
					}
				}
			} else {
				PRINT_E("Invalid tensor type (%d)\n", inputTensorInfo.tensorType);
				return RET_ERR;
			}
		} else {
			PRINT_E("Unsupported data type (%d)\n", inputTensorInfo.dataType);
			return RET_ERR;
		}
	}
	return RET_OK;

}

int32_t InferenceHelperTensorflowLite::invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	if (m_interpreter->Invoke() != kTfLiteOk) {
		PRINT_E("Failed to invoke\n");
		return RET_ERR;
	}
	return RET_OK;
}

void InferenceHelperTensorflowLite::displayModelInfo(const tflite::Interpreter& interpreter)
{
	/* Memo: If you get error around here in Visual Studio, please make sure you don't use Debug */
	const auto& inputIndices = interpreter.inputs();
	int32_t inputNum = static_cast<int32_t>(inputIndices.size());
	PRINT("Input num = %d\n", inputNum);
	for (int32_t i = 0; i < inputNum; i++) {
		auto* tensor = interpreter.tensor(inputIndices[i]);
		PRINT("    tensor[%d]->name: %s\n", i, tensor->name);
		for (int32_t j = 0; j < tensor->dims->size; j++) {
			PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			PRINT("    tensor[%d]->type: quantized\n", i);
			PRINT("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			PRINT("    tensor[%d]->type: not quantized\n", i);
		}
	}

	const auto& outputIndices = interpreter.outputs();
	int32_t outputNum = static_cast<int32_t>(outputIndices.size());
	PRINT("Output num = %d\n", outputNum);
	for (int32_t i = 0; i < outputNum; i++) {
		auto* tensor = interpreter.tensor(outputIndices[i]);
		PRINT("    tensor[%d]->name: %s\n", i, tensor->name);
		for (int32_t j = 0; j < tensor->dims->size; j++) {
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


int32_t InferenceHelperTensorflowLite::getInputTensorInfo(InputTensorInfo& tensorInfo)
{
	for (auto i : m_interpreter->inputs()) {
		TfLiteTensor* tensor = m_interpreter->tensor(i);
		if (std::string(tensor->name) == tensorInfo.name) {
			tensorInfo.id = i;
			
			bool isModelSizeFixed = true;
			for (int32_t i = 0; i < tensor->dims->size; i++) {
				if (tensor->dims->data[i] == -1) isModelSizeFixed = false;
			}
			bool isSizeAssigned = true;
			if ( (tensorInfo.tensorDims.batch == -1) || (tensorInfo.tensorDims.height == -1) || (tensorInfo.tensorDims.width == -1) || (tensorInfo.tensorDims.channel == -1)) isSizeAssigned = false;
			
			if (!isModelSizeFixed && !isSizeAssigned) {
				PRINT_E("Model input size is not set\n");
				return RET_ERR;
			} if (isModelSizeFixed && isSizeAssigned) {
				bool isSizeOK = true;
				for (int32_t i = 0; i < tensor->dims->size; i++) {	// NHWC
					if ((i == 0) && (tensorInfo.tensorDims.batch != tensor->dims->data[0])) isSizeOK = false;
					if ((i == 1) && (tensorInfo.tensorDims.height != tensor->dims->data[1])) isSizeOK = false;
					if ((i == 2) && (tensorInfo.tensorDims.width != tensor->dims->data[2])) isSizeOK = false;
					if ((i == 3) && (tensorInfo.tensorDims.channel != tensor->dims->data[3])) isSizeOK = false;
				}
				if (!isSizeOK) {
					PRINT_E("Invalid size\n");
					return RET_ERR;
				}
			} if (isModelSizeFixed && !isSizeAssigned) {
				PRINT("Input tensor size is set from the model\n");
				tensorInfo.tensorDims.batch = (std::max)(1, tensor->dims->data[0]);
				tensorInfo.tensorDims.height = (std::max)(1, tensor->dims->data[1]);
				tensorInfo.tensorDims.width = (std::max)(1, tensor->dims->data[2]);
				tensorInfo.tensorDims.channel = (std::max)(1, tensor->dims->data[3]);
			} if (!isModelSizeFixed && isSizeAssigned) {
				PRINT("[WARNING] ResizeInputTensor is not tested\n");
				std::vector<int32_t> dims;
				dims.push_back(tensorInfo.tensorDims.batch);
				dims.push_back(tensorInfo.tensorDims.height);
				dims.push_back(tensorInfo.tensorDims.width);
				dims.push_back(tensorInfo.tensorDims.channel);
				m_interpreter->ResizeInputTensor(i, dims);
				if (m_interpreter->AllocateTensors() != kTfLiteOk) {
					PRINT_E("Failed to allocate tensors\n");
					return RET_ERR;
				}
			}

			if (tensor->type == kTfLiteUInt8) tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_UINT8;
			if (tensor->type == kTfLiteFloat32) tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
			if (tensor->type == kTfLiteInt32) tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_INT32;
			if (tensor->type == kTfLiteInt64) tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_INT64;
			return RET_OK;
		}
	}

	PRINT_E("Invalid name (%s) \n", tensorInfo.name.c_str());
	return RET_ERR;
}

int32_t InferenceHelperTensorflowLite::getOutputTensorInfo(OutputTensorInfo& tensorInfo)
{
	for (auto i : m_interpreter->outputs()) {
		const TfLiteTensor* tensor = m_interpreter->tensor(i);
		if (std::string(tensor->name) == tensorInfo.name) {
			tensorInfo.id = i;
			// NHWC
			for (int32_t i = 0; i < tensor->dims->size; i++) {
				if (i == 0) tensorInfo.tensorDims.batch = tensor->dims->data[0];
				if (i == 1) tensorInfo.tensorDims.height = tensor->dims->data[1];
				if (i == 2) tensorInfo.tensorDims.width = tensor->dims->data[2];
				if (i == 3) tensorInfo.tensorDims.channel = tensor->dims->data[3];
			}

			switch (tensor->type) {
			case kTfLiteUInt8:
				tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_UINT8;
				tensorInfo.data = m_interpreter->typed_tensor<uint8_t>(i);
				tensorInfo.quant.scale = tensor->params.scale;
				tensorInfo.quant.zeroPoint = tensor->params.zero_point;
				break;
			case kTfLiteFloat32:
				tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
				tensorInfo.data = m_interpreter->typed_tensor<float>(i);
				break;
			case kTfLiteInt32:
				tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_INT32;
				tensorInfo.data = m_interpreter->typed_tensor<int32_t>(i);
				break;
			case kTfLiteInt64:
				tensorInfo.tensorType = TensorInfo::TENSOR_TYPE_INT64;
				tensorInfo.data = m_interpreter->typed_tensor<int64_t>(i);
				break;
			default:
				return RET_ERR;
			}
			return RET_OK;;
		}
	}
	PRINT_E("Invalid name (%s) \n", tensorInfo.name.c_str());
	return RET_ERR;

}

void InferenceHelperTensorflowLite::convertNormalizeParameters(InputTensorInfo& inputTensorInfo)
{
	if (inputTensorInfo.dataType != InputTensorInfo::DATA_TYPE_IMAGE) return;

#if 0
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
}

static TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src)
{
	if (!src) return nullptr;
	TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(
		malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
	if (!ret) return nullptr;
	ret->size = src->size;
	std::memcpy(ret->data, src->data, src->size * sizeof(float));
	return ret;
}

int32_t InferenceHelperTensorflowLite::setBufferToTensor(int32_t index, void *data)
{
	const TfLiteTensor* tensor = m_interpreter->tensor(index);
	const int32_t modelInputHeight = tensor->dims->data[1];
	const int32_t modelInputWidth = tensor->dims->data[2];
	const int32_t modelInputChannel = tensor->dims->data[3];

	if (tensor->type == kTfLiteUInt8) {
		int32_t dataSize = sizeof(int8_t) * 1 * modelInputHeight * modelInputWidth * modelInputChannel;
		/* Need deep copy quantization parameters */
		/* reference: https://github.com/google-coral/edgetpu/blob/master/src/cpp/basic/basic_engine_native.cc */
		/* todo: do I need to release allocated memory ??? */
		const TfLiteAffineQuantization* inputQuantParams = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
		TfLiteQuantization inputQuantClone;
		inputQuantClone = tensor->quantization;
		TfLiteAffineQuantization* inputQuantParamsClone = reinterpret_cast<TfLiteAffineQuantization*>(malloc(sizeof(TfLiteAffineQuantization)));
		inputQuantParamsClone->scale = TfLiteFloatArrayCopy(inputQuantParams->scale);
		inputQuantParamsClone->zero_point = TfLiteIntArrayCopy(inputQuantParams->zero_point);
		inputQuantParamsClone->quantized_dimension = inputQuantParams->quantized_dimension;
		inputQuantClone.params = inputQuantParamsClone;

		m_interpreter->SetTensorParametersReadOnly(
			index, tensor->type, tensor->name,
			std::vector<int32_t>(tensor->dims->data, tensor->dims->data + tensor->dims->size),
			inputQuantClone,	// use copied parameters
			(const char*)data, dataSize);
	} else {
		int32_t dataSize = sizeof(float) * 1 * modelInputHeight * modelInputWidth * modelInputChannel;
		m_interpreter->SetTensorParametersReadOnly(
			index, tensor->type, tensor->name,
			std::vector<int32_t>(tensor->dims->data, tensor->dims->data + tensor->dims->size),
			tensor->quantization,
			(const char*)data, dataSize);
	}
	return 0;
}

