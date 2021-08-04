#include <jni.h>
#include <string>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "image_processor.h"

//#define WORK_DIR    "/sdcard/resource/"
//#define WORK_DIR    "/mnt/sdcard/resource"
//#define WORK_DIR    "/storage/emulated/0/resource/"
#define WORK_DIR    "/storage/emulated/0/Android/data/com.iwatake.viewandroidtflite/files/Documents/resource"

static std::mutex g_mtx;

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorInitialize(
        JNIEnv* env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    ImageProcessor::InputParam input_param;
    snprintf(input_param.work_dir, sizeof(input_param.work_dir), WORK_DIR);
    input_param.num_threads = 4;
    ret = ImageProcessor::Initialize(input_param);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorProcess(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMat) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    cv::Mat* mat = (cv::Mat*) objMat;
    ImageProcessor::Result result;
    ret = ImageProcessor::Process(*mat, result);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorFinalize(
        JNIEnv* env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    ret = ImageProcessor::Finalize();
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorCommand(
        JNIEnv* env,
        jobject, /* this */
        jint cmd) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    ret = ImageProcessor::Command(cmd);
    return ret;
}
