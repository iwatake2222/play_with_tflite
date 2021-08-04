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

