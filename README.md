# Play with tflite
Sample projects to use Tensorflow Lite for multi-platform

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

- Projects
	- pj_tflite_cls_mobilenet_v2
		- Classification using MobileNet v2
	- pj_tflite_det_mobilenetssd_v1
		- Detection using MobileNetSSD v1
	- pj_tflite_ss_deeplabv3_mnv2
		- Semantic Segmentation using DeepLab v3
	- pj_tflite_hand_mediapipe
		- Palm Detection + Hand Landmark (mediapipe)
	- pj_tflite_arprobe
		- AR-ish application using hand tracking
	- pj_tflite_style_transfer
		- Artistic Style Transfer
	- pj_tflite_pose_movenet
		- Pose detection using MoveNet.SinglePose.Lightning
	- temp_pj_tflite_simple_cls_mobilenet_v2 (Not supported now)
		- Basic project without using InferenceHelper
	- temp_pj_tflite_edgetpuapi_cls_mobilenet_v2 (Not supported now)
		- Basic project using API from coral
	- temp_pj_tflite_edgetpupipeline_cls_inception_v3 (Not supported now)
		- Basic project for Edge TPU Pipeline

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
	cd InferenceHelper/ThirdParty/tensorflow
	chmod +x tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/lite/tools/make/download_dependencies.sh
	```

- Download prebuilt libraries
	- Download prebuilt libraries (ThirdParty.zip) from https://github.com/iwatake2222/InferenceHelper/releases/  (<- Not in this repository)
	- Extract it to `InferenceHelper/ThirdParty/`
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

### Option (Camera input)
```sh
cmake .. -DSPEED_TEST_ONLY=off
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
Pre-built Tensorflow Lite libraries are stored in `InferenceHelper/ThirdParty/tensorflow_prebuilt` . If you want to build them by yourself, please use the following commands.

### Common (Get source code)
```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.4.0
# git checkout 582c8d236cb079023657287c318ff26adb239002  # This is the version I used to generate the libraries
```

### For Linux
You can create libtensorflow.so for x64, armv7 and aarch64 using the following commands in PC Linux like Ubuntu.

- Install tools (Bazel and Python)
	```sh
	sudo apt install bazel
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash ./Miniconda3-latest-Linux-x86_64.sh 
	conda create -n build_tflite
	conda activate build_tflite
	pip install python
	pip install numpy
	```
- Configuration
	```sh
	python configure.py 
	```
- Build
	- Build for x64 Linux
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so \
		-c opt \
		--copt -O3 \
		--strip always \
		--define tflite_with_xnnpack=true

		bazel build  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
		-c opt \
		--copt -DTFLITE_GPU_BINARY_RELEASE \
		--copt -DMESA_EGL_NO_X11_HEADERS \
		--copt -DEGL_NO_X11

		ls bazel-bin/tensorflow/lite/libtensorflowlite.so bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so -la
		```
	- Build for armv7 Linux (option is from build_pip_package_with_bazel.sh)
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so \
		-c opt \
		--config elinux_armhf \
		--copt -march=armv7-a \
		--copt -mfpu=neon-vfpv4 \
		--copt -O3 \
		--copt -fno-tree-pre \
		--copt -fpermissive \
		--define tensorflow_mkldnn_contraction_kernel=0 \
		--define raspberry_pi_with_neon=true \
		--strip always \
		--define tflite_with_xnnpack=true

		bazel build  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
		-c opt \
		--config elinux_armhf \
		--copt -DTFLITE_GPU_BINARY_RELEASE \
		--copt -DMESA_EGL_NO_X11_HEADERS \
		--copt -DEGL_NO_X11
		```
	- Build for aarch64 Linux
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so \
		-c opt \
		--config elinux_aarch64 \
		--define tensorflow_mkldnn_contraction_kernel=0 \
		--copt -O3 \
		--strip always \
		--define tflite_with_xnnpack=true

		bazel build  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
		-c opt \
		--config elinux_aarch64 \
		--copt -DTFLITE_GPU_BINARY_RELEASE \
		--copt -DMESA_EGL_NO_X11_HEADERS \
		--copt -DEGL_NO_X11
		```

### For Windows
You can create libtensorflow.so(it's actually dll) for Windows using the following commands in Windows 10 with Visual Studio 2019. Build with Visual Studio 2017 failed for v2.4.

- Install tools
	- Install an environment providing linux commands (e.g. msys)
		- Add `C:\msys64\usr\bin` to Windows path
	- Install python environment (e.g. miniconda)
	- Install bazel (e.g. locate the bazel executable file into `C:\bin` )
		- Add `C:\bin` to Windows path
		- (I used bazel-3.7.2-windows-x86_64.zip)
- Configuration
	- Open python terminal (e.g. Anaconda Powershell Prompt)
		- (optional) create the new environment
		```sh
		conda create -n build_tflite
		conda activate build_tflite
		pip install python
		pip install numpy
		```
	- Run configuration
		```sh
		cd path-to-tensorflow
		$env:BAZEL_VC="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC"
		python configure.py 
		```
- Build
	- Build for Windows
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so --define tflite_with_xnnpack=true
		# bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
		```
	- The following two library files will be generated:
		- `bazel-bin/tensorflow/lite/libtensorflowlite.so`
		- `bazel-bin/tensorflow/lite/libtensorflowlite.so.if.lib`

### For Android
You can create libtensorflow.so for Android (armv7, aarch64) either in PC Linux or in Windows.
You need to install Android SDK and Android NDK beforehand, then specify the path to sdk and ndk when `python configure.py` asks "`Would you like to interactively configure ./WORKSPACE for Android builds`". (Notice that path must be like `c:/...` instead of `c:\...`)

- Build for armv7 Android
	```sh
	python configure.py 
	bazel build //tensorflow/lite:libtensorflowlite.so `
	-c opt `
	--config android_arm `
	--copt -O3 `
	--strip always `
	--define tflite_with_xnnpack=true

	bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so `
	-c opt `
	--config android_arm `
	--copt -O3 `
	--copt -DTFLITE_GPU_BINARY_RELEASE `
	--strip always 
	```
- Build for aarch64 Android
	```sh
	python configure.py 
	bazel build //tensorflow/lite:libtensorflowlite.so `
	-c opt `
	--config android_arm64 `
	--copt -O3 `
	--strip always `
	--define tflite_with_xnnpack=true

	bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so `
	-c opt `
	--config android_arm64 `
	--copt -O3 `
	--copt -DTFLITE_GPU_BINARY_RELEASE `
	--strip always 
	```
- *Note* : Build with v2.4.0 failed in Windows, but it succeeded in Linux (2021/01/09)
- *Note* : Code for NNAPI delegate is automatically built

## How to create pre-built EdgeTPU library
You can download the official libraries from https://github.com/google-coral/libedgetpu . However, you need to use the same tflite version as used in EdgeTPU library. Since I use v2.4.0 version here, I built my own libraries for EdgeTPU using tensorflow v2.4.0. They are stored in `InferenceHelper/ThirdParty/edgetpu_prebuilt` , and the projects link these libraries.
If you want to build them by yourself, please follow the steps below.

### Common (Get source code)
```sh
git clone https://github.com/google-coral/libedgetpu.git
cd libedgetpu
# git checkout 14eee1a076aa1af7ec1ae3c752be79ae2604a708  # This is the version I used to generate the libraries
```

- Modify setting for bazel to specify tensorflow version
	- Edit `workspace.bzl` file just under the top directory of libedgetpu repo
	- TENSORFLOW_COMMIT: Set the same commit id as tensorflow lite you want to use 
	- TENSORFLOW_SHA256: Calculate SHA256
		- `curl -OL https://github.com/tensorflow/tensorflow/archive/582c8d236cb079023657287c318ff26adb239002.tar.gz`
		- `sha256sum 582c8d236cb079023657287c318ff26adb239002.tar.gz`
	```diff
	$ git diff workspace.bzl
	diff --git a/workspace.bzl b/workspace.bzl
	index 5d05c69..38ca0a7 100644
	--- a/workspace.bzl
	+++ b/workspace.bzl
	@@ -5,8 +5,8 @@ This module contains workspace definitions for building and using libedgetpu.
	load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
	load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

	-TENSORFLOW_COMMIT = "f394a768719a55b5c351ed1ecab2ec6f16f99dd4"
	-TENSORFLOW_SHA256 = "cb286abee7ee9cf5c8701d85fcc88f0fd59e72492ec4f254156de486e3e905c1"
	+TENSORFLOW_COMMIT = "582c8d236cb079023657287c318ff26adb239002"
	+TENSORFLOW_SHA256 = "9c94bfec7214853750c7cacebd079348046f246ec0174d01cd36eda375117628"

	IO_BAZEL_RULES_CLOSURE_COMMIT = "308b05b2419edb5c8ee0471b67a40403df940149"
	IO_BAZEL_RULES_CLOSURE_SHA256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9"
	```


### For Linux
You can create `libedgetpu.so.1.0` for x64, armv7 and aarch64 using the following commands in PC Linux like Ubuntu.

```sh
DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=libedgetpu make docker-build
```

### For Windows
- Download `https://github.com/libusb/libusb/releases/download/v1.0.22/libusb-1.0.22.7z` and extract it the same directory level as libedgetpu. (This is mentioned in `WORKSPACE.windows` )
- Modify `build.bat` to specify BAZEL_VC if needed
	```diff
	$ git diff build.bat
	diff --git a/build.bat b/build.bat
	index 8841c43..bfc8ee5 100644
	--- a/build.bat
	+++ b/build.bat
	@@ -16,8 +16,8 @@ echo off
	setlocal

	if not defined PYTHON set PYTHON=python
	-set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
	-set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
	+set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community
	+set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC
	call "%BAZEL_VC%\Auxiliary\Build\vcvars64.bat"

	for /f %%i in ('%PYTHON% -c "import sys;print(sys.executable)"') do set PYTHON_BIN_PATH=%%i
	```
- Set dumpbin and rc.exe to Windows Path. e.g.:
	- `Set-Item Env:Path "$Env:Path;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64"`
	- `Set-Item Env:Path "$Env:Path;C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64"`
- Resave `driver/usb/usb_driver.cc` as UTF-8 with BOM (You might not need this step)
- Modify `driver/kernel/windows/windows_gasket_ioctl.inc` to avoid compile error (You might not need this step)
	```diff
	$ git diff driver/kernel/windows/windows_gasket_ioctl.inc
	diff --git a/driver/kernel/windows/windows_gasket_ioctl.inc b/driver/kernel/windows/windows_gasket_ioctl.inc
	index 8e6d54e..7cc97cf 100644
	--- a/driver/kernel/windows/windows_gasket_ioctl.inc
	+++ b/driver/kernel/windows/windows_gasket_ioctl.inc
	@@ -10,7 +10,7 @@
	#include "port/fileio.h"
	#include <winioctl.h>
	#endif
	-
	+#include "port/stringprintf.h"
	#ifndef MAX_PATH
	#define MAX_PATH 260
	#endif
	```
- Fix character code issues by following Step 1 ~ Step4 in https://github.com/google/mediapipe/issues/724#issue-622686030
- Run `build.bat` in python terminal like miniconda

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
