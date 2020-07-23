
#ifndef INFERENCE_HELPER_
#define INFERENCE_HELPER_

#include <vector>

class TensorInfo {
public:
	typedef enum {
		TENSOR_TYPE_NONE,
		TENSOR_TYPE_UINT8,
		TENSOR_TYPE_FP32,
		TENSOR_TYPE_INT32,
		TENSOR_TYPE_INT64,
	} TENSOR_TYPE;

public:
	TensorInfo();
	~TensorInfo();
	float* getDataAsFloat();

public:
	int              index;
	TENSOR_TYPE      type;
	void             *data;
	std::vector<int> dims;
	struct {
		float scale;
		int   zeroPoint;
	} quant;

private:
	float  *m_dataFp32;

};

class InferenceHelper {
public:
	typedef enum {
		TENSORFLOW_LITE,
		TENSORFLOW_LITE_EDGETPU,
		TENSORFLOW_LITE_GPU,
		TENSORFLOW_LITE_XNNPACK,
		NCNN,
		NCNN_VULKAN,
		MNN,
		OPEN_CV,
		OPEN_CV_OPENCL,
		TENSOR_RT,
	} HELPER_TYPE;


public:
	virtual ~InferenceHelper() {}
	virtual int initialize(const char *modelFilename, const int numThreads) = 0;
	virtual int finalize(void) = 0;
	virtual int inference(void) = 0;
	virtual int getTensorByName(const char *name, TensorInfo *tensorInfo) = 0;
	virtual int getTensorByIndex(const int index, TensorInfo *tensorInfo) = 0;
	virtual int setBufferToTensorByName(const char *name, const char *data, const unsigned int dataSize) = 0;
	virtual int setBufferToTensorByIndex(const int index, const char *data, const unsigned int dataSize) = 0;
	static InferenceHelper* create(const HELPER_TYPE typeFw);

protected:
	HELPER_TYPE m_helperType;
};

#endif
