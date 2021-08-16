# How to create prebuilt library
- Artifacts:
    - libtensorflowlite.so
    - libtensorflowlite_gpu_delegate.so
    - libedgetpu.so
- Targets:
    - Linux (x64)
    - Linux (armv7, aarch64)
    - Android (armv7, aarch64)

# Preparation
## Requirements
- Ubuntu (18.04 or 20.04)
    - Using 18.04 may be better for compatibility (glibc version)
    - I use Docker in this explanation. The followin docker command is just an example for my environment (command on Windows Power Shell)

```sh
docker pull ubuntu:18.04
docker run -v ${pwd}:/out -e DISPLAY="192.168.1.2:0" -it --name "ubuntu18_tflite_v2.6_build" ubuntu:18.04

# docker pull ubuntu:20.04
# docker run -v ${pwd}:/out -e DISPLAY="192.168.1.2:0" -it --name "ubuntu20_tflite_v2.6_build" ubuntu:20.04
```

The followings are in target OS (Ubuntu 18.04)

## Install tools
```sh
cd ~/
# For TensorFlow Lite
apt update
apt install -y apt-transport-https wget curl gnupg cmake build-essential git unzip
apt install -y mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
apt install -y python python3 python3-pip
pip3 install numpy
wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh
bash ./bazel-3.7.2-installer-linux-x86_64.sh

# For TensorFlow Lite Android
apt install -y --no-install-recommends default-jdk
wget https://dl.google.com/android/repository/commandlinetools-linux-7583922_latest.zip
unzip commandlinetools-linux-7583922_latest.zip
mkdir ~/android-sdk
./cmdline-tools/bin/sdkmanager "platform-tools"  --sdk_root=./android-sdk
./cmdline-tools/bin/sdkmanager "platforms;android-30"  --sdk_root=./android-sdk
./cmdline-tools/bin/sdkmanager "build-tools;30.0.3"  --sdk_root=./android-sdk
# apt install android-sdk
wget https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip
unzip android-ndk-r21e-linux-x86_64.zip

# For Edge TPU
# apt -y install docker.io
```

## Check TENSORFLOW_SHA256 for libedgetpu
- In this tutorial, I will install `v2.6.0`
- TENSORFLOW_COMMIT of `v2.6.0` is `919f693420e35d00c8d0a42100837ae3718f7927`
    - Check on GitHub
- TENSORFLOW_SHA256 is `70a865814b9d773024126a6ce6fea68fefe907b7ae6f9ac7e656613de93abf87`
    - `curl -OL https://github.com/tensorflow/tensorflow/archive/919f693420e35d00c8d0a42100837ae3718f7927.tar.gz`
    - `sha256sum 919f693420e35d00c8d0a42100837ae3718f7927.tar.gz`
- Note the above numbers. I will use them when building libedgetpu later

# Build TensorFlow Lite library
## Get source code
```sh
cd ~/
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.6.0

TFLITE_OUT=/out/tensorflow_prebuilt/
```
## Build for Linux (x64)
- Set default setting during `python3 configure.py`

```sh
python3 configure.py

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

mkdir -p ${TFLITE_OUT}/x64_linux
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ${TFLITE_OUT}/x64_linux/.
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ${TFLITE_OUT}/x64_linux/.
```

## Linux (armv7, aarch64)
- Set default setting during `python3 configure.py`
- Note: Bazel option is from build_pip_package_with_bazel.sh

```sh
python3 configure.py

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

mkdir -p ${TFLITE_OUT}/armv7
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ${TFLITE_OUT}/armv7/.
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ${TFLITE_OUT}/armv7/.

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

mkdir -p ${TFLITE_OUT}/aarch64
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ${TFLITE_OUT}/aarch64/.
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ${TFLITE_OUT}/aarch64/.
```

## Build for Android (armv7, aarch64)
### Modify BUILD script for Android with NNAPI delegate (if needed). This is a workaround. there should be an appropreate option
- `nano tensorflow/lite/BUILD`
    - Add `nnapi:nnapi_delegate` and `nnapi:nnapi_implementation` in `cc_library(name = "framework_experimental",) ...` section
    ``` diff
    diff --git a/tensorflow/lite/BUILD b/tensorflow/lite/BUILD
    index 72a46e6675f..6587b2ddfa9 100644
    --- a/tensorflow/lite/BUILD
    +++ b/tensorflow/lite/BUILD
    @@ -353,6 +353,8 @@ cc_library(
            "//tensorflow/lite/core/api:verifier",
            "//tensorflow/lite/experimental/resource",
            "//tensorflow/lite/schema:schema_fbs",
    +        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
    +        "//tensorflow/lite/nnapi:nnapi_implementation",
            "@flatbuffers//:runtime_cc",
        ],
        alwayslink = 1,  # TODO(b/161243354): eliminate this.
    ```

### Build
- Set Android SDK and Android NDK path setting during `python3 configure.py`
    - Answer yes to "`Would you like to interactively configure ./WORKSPACE for Android builds`"
        - `/root/android-ndk-r21e`
        - `/root/android-sdk`

```sh
python3 configure.py

bazel build //tensorflow/lite:libtensorflowlite.so \
-c opt \
--config android_arm \
--copt -O3 \
--strip always \
--define tflite_with_xnnpack=true

bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
-c opt \
--config android_arm \
--copt -O3 \
--copt -DTFLITE_GPU_BINARY_RELEASE \
--strip always 

mkdir -p ${TFLITE_OUT}/android/armeabi-v7a
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ${TFLITE_OUT}/android/armeabi-v7a/.
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ${TFLITE_OUT}/android/armeabi-v7a/.


bazel build //tensorflow/lite:libtensorflowlite.so \
-c opt \
--config android_arm64 \
--copt -O3 \
--strip always \
--define tflite_with_xnnpack=true

bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
-c opt \
--config android_arm64 \
--copt -O3 \
--copt -DTFLITE_GPU_BINARY_RELEASE \
--strip always 

mkdir -p ${TFLITE_OUT}/android/arm64-v8a
cp bazel-bin/tensorflow/lite/libtensorflowlite.so ${TFLITE_OUT}/android/arm64-v8a/.
cp bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so ${TFLITE_OUT}/android/arm64-v8a/.
```


# Build EdgeTPU library
If you use Ubuntu on Docker, the following process may fail (docker on docker).
I run the following commands on WSL2.

## Get source code
```sh
git clone https://github.com/google-coral/libedgetpu.git
cd libedgetpu
git checkout ea1eaddbddece0c9ca1166e868f8fd
```

## Modify commit id
```sh
sed -i s/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/919f693420e35d00c8d0a42100837ae3718f7927/g workspace.bzl
sed -i s/cb99f136dc5c89143669888a44bfdd134c086e1e2d9e36278c1eb0f03fe62d76/70a865814b9d773024126a6ce6fea68fefe907b7ae6f9ac7e656613de93abf87/g workspace.bzl
```

## Build for Linux (x64, armv7, aarch64)
```sh
# DOCKER_CPUS="k8" DOCKER_IMAGE="ubuntu:18.04" DOCKER_TARGETS=libedgetpu make docker-build
# DOCKER_CPUS="armv7a aarch64" DOCKER_IMAGE="debian:stretch" DOCKER_TARGETS=libedgetpu make docker-build
DOCKER_CPUS="k8 armv7a aarch64" DOCKER_IMAGE="ubuntu:18.04" DOCKER_TARGETS=libedgetpu make docker-build
```
