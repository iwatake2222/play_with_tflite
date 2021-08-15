# How to create prebuilt library
- Artifacts:
    - libtensorflowlite.so
    - libedgetpu.so
- Targets:
    - Windows (x64)

## Requirements
- Windows 10 64-bit
- Visual Studio 2019 (Community)
- msys
- Python3
    - I use pyenv and venv

## Install tools
- Install msys ( https://www.msys2.org )
    - https://github.com/msys2/msys2-installer/releases/download/2021-07-25/msys2-x86_64-20210725.exe
    - Add `C:\msys64\usr\bin` to path

- Install Git in msys
    - `pacman -S git`

- Install bazel
    - https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-windows-x86_64.zip
    - Extract to `C:\bin\bazel.exe`
    - Add `C:\bin` to path

- If you have added GitBash to path, remove it 

## Build TensorFlow Lite library
### Get source code
```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.6.0
```

### Build for Windows
- Open PowerShell
- Set default setting during `python3 configure.py`

```sh
pyenv shell 3.8.9
python -m venv C:\iwatake\devel\python\.venv\py38_tflite_build_01
C:\iwatake\devel\python\.venv\py38_tflite_build_01\Scripts\Activate.ps1
pip install numpy

cd path-to-tensorflow
$env:BAZEL_VC="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC"
python configure.py
bazel build //tensorflow/lite:libtensorflowlite.so --define tflite_with_xnnpack=true
# bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

mkdir x64_windows
cp bazel-bin/tensorflow/lite/libtensorflowlite.so x64_windows/.
cp bazel-bin/tensorflow/lite/libtensorflowlite.so.if.lib x64_windows/.
```


## Build EdgeTPU library
### Get source code
```sh
git clone https://github.com/google-coral/libedgetpu.git
cd libedgetpu
git checkout ea1eaddbddece0c9ca1166e868f8fd
```

### Modify commit id
```sh
sed -i s/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/919f693420e35d00c8d0a42100837ae3718f7927/g workspace.bzl
sed -i s/cb99f136dc5c89143669888a44bfdd134c086e1e2d9e36278c1eb0f03fe62d76/70a865814b9d773024126a6ce6fea68fefe907b7ae6f9ac7e656613de93abf87/g workspace.bzl
```

### Download libusb
- https://github.com/libusb/libusb/releases/download/v1.0.24/libusb-1.0.24.7z
- Place libusb/ at the same directory as libedgetpu/
    ```
    xxx/
       - libusb/    <- new
       - libedgetpu/
    ```

### Workaround
- Resave `driver/usb/usb_driver.cc` as UTF-8 with BOM (You may not need this step)

### Workaround
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

### Build for Windows
- Open Visual Studio 2019 Developer Command Prompt

```sh
cd path-to-libedgetpu
C:\iwatake\devel\python\.venv\py38_tflite_build_01\Scripts\activate.bat
set BAZEL_VC="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC"
build.bat
```

### For runtime
- Install `edgetpu_runtime_20210119.zip`
    - Execution failed with `edgetpu_runtime_20210726.zip` for some reasons
    - If you have already installed `edgetpu_runtime_20210726.zip` , uninstall it. Also uninstall `UsbDk Runtime Libraries` from windows
- Delete `C:\Windows\System32\edgetpu.dll` so that your project uses the created edgetpu.dll
    - or copy the created edgetpu.dll to `C:\Windows\System32\edgetpu.dll`
