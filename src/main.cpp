#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

// Choose target shape (320x320, 320x240, or 240x320) based on original aspect ratio
static cv::Size choose_target_shape(const cv::Size& orig) {
    double ar = double(orig.height) / double(orig.width);
    auto d = [](double a, double b) { return std::abs(a - b); };
    double square = d(ar, 1.0);
    double landscape = d(ar, double(240) / 320);
    double portrait = d(ar, double(320) / 240);
    if (square <= landscape && square <= portrait) return {320, 320};
    if (landscape <= portrait) return {320, 240};  // (w, h)
    return {240, 320};
}

// Letterbox structure to hold preprocessed tensor and padding info
struct Letterbox {
    cv::Mat tensor;  // HxWxC float32, RGB, 0..255
    int top = 0, bottom = 0, left = 0, right = 0;
};

// Preprocess BGR image: convert to RGB, resize with aspect ratio, pad to target shape
static Letterbox preprocess_bgr(const cv::Mat& img_bgr) {
    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
    img_rgb.convertTo(img_rgb, CV_32F);  // keep 0..255 float (no /255)

    const cv::Size orig = img_rgb.size();
    const cv::Size target = choose_target_shape(orig);

    const double scale = std::min(double(target.width) / orig.width,
                                  double(target.height) / orig.height);
    cv::Size scaled(int(std::round(orig.width * scale)),
                    int(std::round(orig.height * scale)));

    cv::Mat resized;
    cv::resize(img_rgb, resized, scaled, 0, 0, cv::INTER_LINEAR);

    int vertical = target.height - resized.rows;
    int horizontal = target.width - resized.cols;

    int top = vertical / 2, bottom = vertical - top;
    int left = horizontal / 2, right = horizontal - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return {padded, top, bottom, left, right};
}

// Postprocess output saliency map: crop padding and resize to original size
static cv::Mat postprocess_to_original(const cv::Mat& out_map, int top, int bottom,
                                       int left, int right, const cv::Size& orig) {
    int H = out_map.rows, W = out_map.cols;
    int y = std::max(0, top);
    int x = std::max(0, left);
    int h = std::max(1, H - top - bottom);
    int w = std::max(1, W - left - right);
    cv::Rect roi(x, y, w, h);
    roi &= cv::Rect(0, 0, W, H);

    cv::Mat cropped = out_map(roi).clone();
    cv::Mat restored;
    cv::resize(cropped, restored, orig, 0, 0, cv::INTER_LINEAR);
    return restored;  // HxW (float32)
}

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "models/msi_net.onnx";
    std::string image_path = (argc > 2) ? argv[2] : "assets/example.jpg";

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 1;
    }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Image not found: " << image_path << std::endl;
        return 1;
    }

    // --- Preprocess ---
    auto lb = preprocess_bgr(img);
    cv::Mat input = lb.tensor;  // HxWx3 float32 RGB

    // --- ONNX Runtime setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "msi");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency()));
    // (CPU EP by default; CoreML EP can be added later)

    Ort::Session session(env, model_path.c_str(), so);

    // Fetch I/O names (new ORT API returns smart pointers)
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_in = session.GetInputCount();
    size_t num_out = session.GetOutputCount();
    if (num_in != 1 || num_out != 1) {
        std::cerr << "Unexpected I/O counts (inputs=" << num_in
                  << ", outputs=" << num_out << ")\n";
        return 1;
    }
    Ort::AllocatedStringPtr in_name_ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr out_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string input_name = in_name_ptr ? std::string(in_name_ptr.get()) : "input";
    std::string output_name = out_name_ptr ? std::string(out_name_ptr.get()) : "output";

    // Expect NHWC float32 (TF export). Keep HWC memory layout from OpenCV.
    std::vector<int64_t> input_shape = {1, input.rows, input.cols, 3};
    const size_t numel = static_cast<size_t>(input.total()) * static_cast<size_t>(input.channels());
    std::vector<float> nhwc(numel);
    std::memcpy(nhwc.data(), input.ptr<float>(), numel * sizeof(float));

    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(mem, nhwc.data(), nhwc.size(),
                                        input_shape.data(), input_shape.size());

    const char* in_names[] = {input_name.c_str()};
    const char* out_names[] = {output_name.c_str()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    // Output: [1, H, W, 1] float32
    float* out_data = outputs[0].GetTensorMutableData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_dims = out_info.GetShape();
    if (out_dims.size() != 4 || out_dims[0] != 1 || out_dims[3] != 1) {
        std::cerr << "Unexpected output shape\n";
        return 1;
    }
    int H = static_cast<int>(out_dims[1]);
    int W = static_cast<int>(out_dims[2]);
    cv::Mat out_map(H, W, CV_32FC1, out_data);

    // Remove letterbox padding and resize back to original
    cv::Mat sal = postprocess_to_original(out_map, lb.top, lb.bottom, lb.left, lb.right, img.size());

    cv::Mat sal_u8;
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(sal, &mn, &mx);
    if (mx > mn) {
        cv::Mat tmp32f;
        sal.convertTo(tmp32f, CV_32F);  // ensure float32

        // scale to [0,255]
        tmp32f = (tmp32f - static_cast<float>(mn)) *
                (255.0f / static_cast<float>(mx - mn));

        // +0.5 for round-to-nearest, then clamp to [0,255]
        cv::add(tmp32f, 0.5, tmp32f);
        cv::min(tmp32f, 255.0, tmp32f);
        cv::max(tmp32f, 0.0, tmp32f);

        // floor element-wise (use std::floor since cv::floor may be unavailable)
        cv::Mat floored32f(tmp32f.size(), CV_32F);
        for (int r = 0; r < tmp32f.rows; ++r) {
            const float* src = tmp32f.ptr<float>(r);
            float* dst = floored32f.ptr<float>(r);
            for (int c = 0; c < tmp32f.cols; ++c) {
                dst[c] = std::floor(src[c]);
            }
        }

        // cast to uint8 (no additional rounding)
        floored32f.convertTo(sal_u8, CV_8U);
    } else {
        sal_u8 = cv::Mat::zeros(sal.size(), CV_8U);
    }

    // Save deterministic outputs for comparison
    cv::imwrite("onnx_result_u8.png", sal_u8);     // for byte-perfect PNG compare
    // Optional: save raw float saliency (EXR keeps 32F)
    // cv::imwrite("onnx_result.exr", sal);

    // Optional visualization (pretty overlay)
    cv::Mat color, blended;
    cv::applyColorMap(sal_u8, color, cv::COLORMAP_INFERNO);
    cv::addWeighted(color, 0.65, img, 0.35, 0.0, blended);

    // cv::imshow("Input", img);
    // cv::imshow("Saliency", blended);
    // cv::waitKey(0);
    cv::imwrite("onnx_result_overlay.png", blended);

    return 0;
}
