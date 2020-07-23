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
	- Android (armv7)
		- not tested yet
	- Android (aarch64)
		- not tested yet
	- Windows (x64). Visual Studio 2017
		- Tested in Windows10 64-bit
	- (Only native build is supported)

- Delegate
	- Edge TPU
		- Tested in Raspberry Pi (armv7) and Jetson NX (aarch64)
	- XNNPACK
		- Tested in Jetson NX
	- GPU
		- Tested in Jetson NX

## How to build application
### Common 
- Get source code
	```sh
	git clone https://github.com/iwatake2222/play_with_tflite.git
	cd play_with_tflite

	git submodule init
	git submodule update
	cd third_party/tensorflow
	chmod +x tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/lite/tools/make/download_dependencies.sh
	```

- Download prebuilt libraries and models
	- Download prebuilt libraries (third_party.zip) and models (resource.zip) from https://github.com/iwatake2222/play_with_tflite/releases/ .
Extract them to `third_party` and `resource`

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
	- `Where is the source code` : path-to-play_with_tflite/pj_tflite_cls_mobilenet_v2	(for example)
	- `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

**Note**
I found running with `Debug` causes exception, so use `Release` or `RelWithDebInfo` in Visual Studio.

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
cmake .. -DTFLITE_DELEGATE_EDGETPU=on  -DTFLITE_DELEGATE_GPU=off -DTFLITE_DELEGATE_XNNPACK=off
cp libedgetpu.so.1.0 libedgetpu.so.1
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
sudo LD_LIBRARY_PATH=./ ./main

# GPU
cmake .. -DTFLITE_DELEGATE_EDGETPU=off -DTFLITE_DELEGATE_GPU=on  -DTFLITE_DELEGATE_XNNPACK=off
# you may need `sudo apt install ocl-icd-opencl-dev` or `sudo apt install libgles2-mesa-dev`

# XNNPACK
cmake .. -DTFLITE_DELEGATE_EDGETPU=off -DTFLITE_DELEGATE_GPU=off -DTFLITE_DELEGATE_XNNPACK=on
```

## How to create pre-built TensorflowLite library
Pre-built TensorflowLite libraries are stored in `third_party/tensorflow_prebuilt` . If you want to build them by yourself, please use the following commands.

### Common (Get source code)
```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r2.3
# git checkout bb3c460114b13fda5c730fe43587b8e8c2243cd7  # This is the version I used to generate the libraries
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
You can create libtensorflow.so(it's actually dll) for Windows using the following commands in Windows 10 with Visual Studio. I used Visual Studio 2017 (You don't need to specify toolchain path. Bazel will automatically find it).

- Modify setting for bazel (workaround)
	- Reference: https://github.com/tensorflow/tensorflow/issues/28824#issuecomment-536669038
	- Edit `WORKSPACE` file just under the top directory of tensorflow repo
	```bazel
	$ git diff
	diff --git a/WORKSPACE b/WORKSPACE
	index ea741c31c7..2115267603 100644
	--- a/WORKSPACE
	+++ b/WORKSPACE
	@@ -12,6 +12,13 @@ http_archive(
		],
	)

	+http_archive(
	+    name = "io_bazel_rules_docker",
	+    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
	+    strip_prefix = "rules_docker-0.7.0",
	+    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"],
	+)
	+
	# Load tf_repositories() before loading dependencies for other repository so
	# that dependencies like com_google_protobuf won't be overridden.
	load("//tensorflow:workspace.bzl", "tf_repositories")
	```

- Install tools
	- Install an environment providing linux commands (e.g. msys)
		- Add `C:\msys64\usr\bin` to Windows path
	- Install python environment (e.g. miniconda)
	- Install bazel (e.g. locate the bazel executable file into `C:\bin` )
		- Add `C:\bin` to Windows path
		- (I used bazel-3.4.1-windows-x86_64.zip)
- Configuration
	- Open python terminal (e.g. Anaconda Powershell Prompt)
		- (optional) create the new environment
		```sh
		conda create -n build_tflite
		conda activate build_tflite
		pip install python
		pip install numpy
		```
	```sh
	cd path-to-tensorflow
	python configure.py 
	```
- Build
	- Build for Windows
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so `
		-c opt `
		--copt -O3 `
		--strip always  `
		

		# bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so `
		# -c opt `
		# --copt -O3 `
		# --copt -DTFLITE_GPU_BINARY_RELEASE `
		# --strip always 
		```

### [WIP] For Windows with XNNPACK
If you want to build Tensorflow Lite with XNNPACK, you need some more works.
(When I just add `--define tflite_with_xnnpack=true` option, I got link error for `_cvtu32_mask16` ... it looks, I need to use LLVM to build)

- Install LLVM Compiler Toolchain for Visual Studio
	- https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.llvm-toolchain
- Install LLVM
	- https://releases.llvm.org/download.html
	- (https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/LLVM-10.0.0-win64.exe ) I used this.
- Modify bazel to build with CLang (just follow the instructions here (https://docs.bazel.build/versions/master/windows.html#build-c-with-clang))
	- add the following in the top level BUILD file
	```bazel
	platform(
		name = "x64_windows-clang-cl",
		constraint_values = [
			"@platforms//cpu:x86_64",
			"@platforms//os:windows",
			"@bazel_tools//tools/cpp:clang-cl",
		],
	)
	```
- Build
```sh
set BAZEL_LLVM=C:\Program Files\LLVM
bazel build //tensorflow/lite:libtensorflowlite.so `
-c opt `
--copt -O3 `
--strip always `
--define tflite_with_xnnpack=true `
--extra_toolchains=@local_config_cc//:cc-toolchain-x64_windows-clang-cl --extra_execution_platforms=//:x64_windows-clang-cl `
--incompatible_enable_cc_toolchain_resolution
```

### For Android
You can create libtensorflow.so for Android (armv7, aarch64) both in PC Linux and in Windows. I used Windows 10 to build.
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

## How to create pre-built EdgeTPU library
The official libraries for EdgeTPU are stored in `third_party/edgetpu/libedgetpu` . However, you need to use the same tflite version as used in EdgeTPU library build. Since I use r2.3 branch here, I built my own libraries for EdgeTPU using tensorflow r2.3. They are stored in `third_party/edgetpu_prebuilt` , and the projects link these libraries.
If you want to build them by yourself, please follow the steps below.

### Common (Get source code)
```sh
git clone https://github.com/google-coral/libedgetpu.git
cd libedgetpu
# git checkout f8cac1044e3ca32b6a9c8712ac6d063e58f19fe1  # This is the version I used to generate the libraries
```

- Modify setting for bazel to specify tensorflow version
	- Edit `WORKSPACE` file just under the top directory of tensorflow repo
	- The version must be the same as tensorflow lite you want to use
	```
	diff --git a/bazel/WORKSPACE b/bazel/WORKSPACE
	index f9cb049..ec735c4 100644
	--- a/bazel/WORKSPACE
	+++ b/bazel/WORKSPACE
	@@ -36,9 +36,9 @@ http_archive(
	# repo WORKSPACE file.
	# TODO: figure out a way to keep single source of truth of the
	# TF commit # used.
	-TENSORFLOW_COMMIT = "f394a768719a55b5c351ed1ecab2ec6f16f99dd4";
	+TENSORFLOW_COMMIT = "bb3c460114b13fda5c730fe43587b8e8c2243cd7";
	# Command to calculate: curl -OL <FILE-URL> | sha256sum | awk '{print $1}'
	-TENSORFLOW_SHA256 = "cb286abee7ee9cf5c8701d85fcc88f0fd59e72492ec4f254156de486e3e905c1"
	+TENSORFLOW_SHA256 = "1d358199ea52d38524311dee2fb8f08a5c4c444bd0fcd8a1fe2209cac47afffb"
	http_archive(
		name = "org_tensorflow",
		sha256 = TENSORFLOW_SHA256,
	```


### For Linux
You can create `libedgetpu.so.1.0` for x64, armv7 and aarch64 using the following commands in PC Linux like Ubuntu.

```sh
DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=libedgetpu make docker-build
```

### For Windows (WIP)
- Download `https://github.com/libusb/libusb/releases/download/v1.0.22/libusb-1.0.22.7z` and extract it the same directory level as libedgetpu. (This is mentioned in `WORKSPACE.windows` )
- Resave `usb_driver.cc` as UTF-8 with BOM (???)
- Set dumpbin to Windows Path
	- e.g.: `Set-Item Env:Path "$Env:Path;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"`
- Run `build.bat` in python terminal like miniconda
- Build fails so far. If I don't change TENSORFLOW_COMMIT, everything is okay thuogh...

## Acknowledgements
- References:
	- https://www.tensorflow.org/lite/performance/gpu_advanced
	- https://www.tensorflow.org/lite/guide/android
	- https://qiita.com/terryky/items/fa18bd10cfead076b39f
	- https://github.com/terryky/tflite_gles_app

- This project includes output files (such as `libtensorflowlite.so`) of the following project:
	- https://github.com/tensorflow/tensorflow
- This project includes models:
	- mobilenetv2-1.0
		- https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
		- https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
		- http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
		- https://coral.withgoogle.com/models/
		- https://www.tensorflow.org/lite/guide/hosted_models
	- coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		- https://www.tensorflow.org/lite/models/object_detection/overview
		- https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
	- deeplabv3_mnv2_dm05_pascal_quant
		- https://github.com/google-coral/edgetpu/tree/master/test_data
		- https://github.com/google-coral/edgetpu/blob/master/test_data/deeplabv3_mnv2_dm05_pascal_quant.tflite
		- https://github.com/google-coral/edgetpu/blob/master/test_data/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite
