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
