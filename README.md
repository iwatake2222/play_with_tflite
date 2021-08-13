# Play with tflite
- Sample projects to use TensorFlow Lite in C++ for multi-platform
- Typical project structure is like the following diagram
    - ![00_doc/design.jpg](00_doc/design.jpg)

## Target Environment
- Platform
    - Linux (x64)
        - Tested in Xubuntu 18 in VirtualBox in Windows 10
    - Linux (armv7)
        - Tested in Raspberry Pi4 (Raspbian 32-bit)
    - Linux (aarch64)
        - Tested in Jetson Nano (JetPack 4.3) and Jetson NX (JetPack 4.4)
    - Android (aarch64)
        - Tested in Pixel 4a
    - Windows (x64). Visual Studio 2017, 2019
        - Tested in Windows10 64-bit

- Delegate
    - Edge TPU
        - Tested in Windows, Raspberry Pi (armv7) and Jetson NX (aarch64)
    - XNNPACK
        - Tested in Windows, Raspberry Pi (armv7) and Jetson NX (aarch64)
    - GPU
        - Tested in Jetson NX and Android
    - NNAPI(CPU, GPU, DSP)
        - Tested in Android (Pixel 4a)

## Usage
```
./main [input]

 - input = blank
    - use the default image file set in source code (main.cpp)
    - e.g. ./main
 - input = *.mp4, *.avi, *.webm
    - use video file
    - e.g. ./main test.mp4
 - input = *.jpg, *.png, *.bmp
    - use image file
    - e.g. ./main test.jpg
 - input = number (e.g. 0, 1, 2, ...)
    - use camera
    - e.g. ./main 0
```

## How to build application
### Requirements
- OpenCV 4.x
- Edge TPU runtime if needed
    - https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime
    - https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20201204.zip

### Common 
- Get source code
    ```sh
    git clone https://github.com/iwatake2222/play_with_tflite.git
    cd play_with_tflite

    git submodule update --init --recursive
    cd InferenceHelper/third_party/tensorflow
    chmod +x tensorflow/lite/tools/make/download_dependencies.sh
    tensorflow/lite/tools/make/download_dependencies.sh
    ```

- Download prebuilt libraries
    - Download prebuilt libraries (third_party.zip) from https://github.com/iwatake2222/InferenceHelper/releases/  (<- Not in this repository)
    - Extract it to `InferenceHelper/third_party/`
- Download models
    - Download models (resource.zip) from https://github.com/iwatake2222/play_with_tflite/releases/ 
    - Extract it to `resource/`

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
    - `Where is the source code` : path-to-play_with_tflite/pj_tflite_cls_mobilenet_v2	(for example)
    - `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

**Note**
Running with `Debug` causes exception, so use `Release` or `RelWithDebInfo` in Visual Studio.

### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```sh
cd pj_tflite_cls_mobilenet_v2   # for example
mkdir build && cd build
cmake ..
make
./main
```

### Options (Delegate)
```sh
# Edge TPU
cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU=on  -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK=off
cp libedgetpu.so.1.0 libedgetpu.so.1
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
sudo LD_LIBRARY_PATH=./ ./main
# you may get "Segmentation fault (core dumped)" without sudo

# GPU
cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU=on  -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK=off
# you may need `sudo apt install ocl-icd-opencl-dev` or `sudo apt install libgles2-mesa-dev`

# XNNPACK
cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK=on

# NNAPI (Note: You use Android for NNAPI. Therefore, you will modify CMakeLists.txt in Android Studio rather than the following command)
cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK=off -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI=on
```

You also need to select framework when calling `InferenceHelper::create` .

### Android
- Requirements
    - Android Studio
        - Compile Sdk Version
            - 30
        - Build Tools version
            - 30.0.0
        - Target SDK Version
            - 30
        - Min SDK Version
            - 24
            - With 23, I got the following error
                - `bionic/libc/include/bits/fortify/unistd.h:174: undefined reference to `__write_chk'`
                - https://github.com/android/ndk/issues/1179
    - Android NDK
        - 21.3.6528147
    - OpenCV
        - opencv-4.4.0-android-sdk.zip
    - *The version is just the version I used


- Configure NDK
    - File -> Project Structure -> SDK Location -> Android NDK location
        - C:\Users\abc\AppData\Local\Android\Sdk\ndk\21.3.6528147
- Import OpenCV
    - Download and extract OpenCV android-sdk (https://github.com/opencv/opencv/releases )
    - File -> New -> Import Module
        - path-to-opencv\opencv-4.3.0-android-sdk\OpenCV-android-sdk\sdk
    - FIle -> Project Structure -> Dependencies -> app -> Declared Dependencies -> + -> Module Dependencies
        - select sdk
    - In case you cannot import OpenCV module, remove sdk module and dependency of app to sdk in Project Structure
        - Do `git update-index --skip-worktree ViewAndroid/app/build.gradle ViewAndroid/settings.gradle ViewAndroid/.idea/gradle.xml` not to save modified settings including opencv sdk
- Modify `ViewAndroid\app\src\main\cpp\CMakeLists.txt` to call image processor function you want to use.
    - `set(ImageProcessor_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../pj_tflite_arprobe/ImageProcessor")`
- Copy `resource` directory to `/storage/emulated/0/Android/data/com.iwatake.viewandroidtflite/files/Documents/resource` (<- e.g.) . The directory will be created after running the app (so the first run should fail because model files cannot be read)

- *Note* : By default, `InferenceHelper::TENSORFLOW_LITE` is used. You can modify `ViewAndroid\app\src\main\cpp\CMakeLists.txt` to select which delegate to use. It's better to use `InferenceHelper::TENSORFLOW_LITE_GPU` to get high performance.


### NNAPI
By default, NNAPI will select the most appropreate accelerator for the model. You can specify which accelerator to use by yourself. Modify the following code in `InferenceHelperTensorflowLite.cpp`

```
// options.accelerator_name = "qti-default";
// options.accelerator_name = "qti-dsp";
// options.accelerator_name = "qti-gpu";
```


## How to create pre-built Tensorflow Lite library
Pre-built Tensorflow Lite libraries are stored in `InferenceHelper/ThirdParty/tensorflow_prebuilt` . If you want to build them by yourself, please use the following instruction

[00_doc/how_to_create_prebuilt_tensorflow_lite_library_v2.4.0.md](00_doc/how_to_create_prebuilt_tensorflow_lite_library_v2.4.0.md)

## How to create pre-built EdgeTPU library
You can download the official libraries from https://github.com/google-coral/libedgetpu . However, you need to use the same tflite version as used in EdgeTPU library. Since I use v2.4.0 version here, I built my own libraries for EdgeTPU using tensorflow v2.4.0. They are stored in `InferenceHelper/ThirdParty/edgetpu_prebuilt` , and the projects link these libraries.
If you want to build them by yourself, please use the following instruction.

[00_doc/how_to_create_prebuilt_edgetpu_library_v2.4.0.md](00_doc/how_to_create_prebuilt_edgetpu_library_v2.4.0.md)

# License
- play_with_tflite
- https://github.com/iwatake2222/play_with_tflite
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0

# Acknowledgements
- This project utilizes OSS (Open Source Software)
    - [NOTICE.md](NOTICE.md)
- This project utilizes models from other projects:
    - Please find `model_information.md` in resource.zip
