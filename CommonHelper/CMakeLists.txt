cmake_minimum_required(VERSION 3.0)

set(LibraryName "CommonHelper")

# Create library
set(SRC CommonHelper.h CommonHelper.cpp)
add_library(${LibraryName} ${SRC})
