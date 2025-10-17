cmake_minimum_required(VERSION 3.20)
project(msi_macinference CXX)
set(CMAKE_CXX_STANDARD 17)

# Homebrew prefix
execute_process(COMMAND brew --prefix OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(OpenCV REQUIRED)

# ---- ONNX Runtime ----
set(ONNXRUNTIME_INCLUDE_DIR "${BREW_PREFIX}/include/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime HINTS "${BREW_PREFIX}/lib" REQUIRED)

add_executable(msi src/main.cpp)
target_include_directories(msi PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(msi PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})