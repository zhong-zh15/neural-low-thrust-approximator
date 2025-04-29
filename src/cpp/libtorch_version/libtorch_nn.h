#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "eigen_nn_csvreader.h"
#include "nn_input_map.h"

// ==============================
// 5. Helper function: Read scaler parameter CSV file
//    Assumes first row is header, subsequent rows contain mean, scale, var
//    Returns pair: (means, scales)
// ==============================
inline std::pair<std::vector<double>, std::vector<double>> read_scaler_csv(const std::string& filename) {
    std::vector<double> means;
    std::vector<double> scales;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to open scaler file: " << filename << std::endl;
        return { means, scales };
    }
    std::string line;
    if (!std::getline(infile, line)) {
        std::cerr << "Scaler file is empty: " << filename << std::endl;
        return { means, scales };
    }
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string mean_str, scale_str, var_str;
        if (std::getline(ss, mean_str, ',') &&
            std::getline(ss, scale_str, ',') &&
            std::getline(ss, var_str, ',')) {
            try {
                double m = std::stod(mean_str);
                double s = std::stod(scale_str);
                means.push_back(m);
                scales.push_back(s);
            }
            catch (...) {
                // Skip conversion errors
            }
        }
    }
    infile.close();
    return { means, scales };
}

// ==============================
// 6. Convert 2D vector to torch::Tensor (float32)
// ==============================
inline torch::Tensor tensor_from_vector(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return torch::empty({ 0 });
    size_t rows = data.size();
    size_t cols = data[0].size();
    std::vector<float> flat;
    flat.reserve(rows * cols);
    for (const auto& row : data) {
        for (double val : row) {
            flat.push_back(static_cast<float>(val));
        }
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(flat.data(), { static_cast<long>(rows), static_cast<long>(cols) }, options).clone();
    return tensor;
}

// ==============================
// 8. Overloaded fast_predict_vector interfaces
//    Version 1: Batch prediction with 2D vector<double> input (one sample per row)
//    Version 2: Single-sample prediction with 1D vector<double>
//    Performs data conversion, normalization, single-thread inference, and denormalization
//    Uses thread-local static buffers and preallocated Tensors
// ==============================

// Overload version 1: Batch prediction
inline std::vector<double> fast_predict_vector(
    std::vector<std::vector<double>> raw_data,
    const torch::Tensor& X_mean,
    const torch::Tensor& X_scale,
    const torch::Tensor& Y_mean,
    const torch::Tensor& Y_scale,
    torch::jit::script::Module& module,
    const int type = 1)
{
    std::vector<double> v0_norm_vector;
    v0_norm_vector.reserve(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); ++i) {
        double v0_norm = 0.0;
        if (type == 1)
            raw_data[i] = nn_input_2_normlization_dv(raw_data[i], v0_norm);
        else
            raw_data[i] = nn_input_2_normlization_tmin(raw_data[i], v0_norm);
        v0_norm_vector.push_back(v0_norm);
    }

    size_t sample_count = raw_data.size();
    if (sample_count == 0) return {};
    size_t feature_count = raw_data[0].size();

    thread_local std::vector<float> buffer;
    buffer.resize(sample_count * feature_count);
    for (size_t i = 0; i < sample_count; ++i) {
        for (size_t j = 0; j < feature_count; ++j) {
            buffer[i * feature_count + j] = static_cast<float>(raw_data[i][j]);
        }
    }
    auto X_tensor = torch::from_blob(buffer.data(), { static_cast<long>(sample_count), static_cast<long>(feature_count) }, torch::kFloat32).clone();
    auto X_norm = (X_tensor - X_mean) / X_scale;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(X_norm);

    int old_threads = torch::get_num_threads();
    torch::set_num_threads(1);
    torch::NoGradGuard no_grad;
    auto preds_norm = module.forward(inputs).toTensor();
    torch::set_num_threads(old_threads);

    auto preds_orig = preds_norm * Y_scale + Y_mean;
    long n = preds_orig.size(0);
    std::vector<double> output(n);
    auto accessor = preds_orig.accessor<float, 2>();
    for (long i = 0; i < n; ++i) {
        output[i] = static_cast<double>(accessor[i][0]) * v0_norm_vector[i];
    }
    return output;
}

// Overload version 2: Single-sample prediction
inline double fast_predict_vector(
    std::vector<double> raw_data,
    const torch::Tensor& X_mean,
    const torch::Tensor& X_scale,
    const torch::Tensor& Y_mean,
    const torch::Tensor& Y_scale,
    torch::jit::script::Module& module,
    const int type = 1)
{
    double v0_norm = 0.0;
    if (type == 1)
        raw_data = nn_input_2_normlization_dv(raw_data, v0_norm);
    else
        raw_data = nn_input_2_normlization_tmin(raw_data, v0_norm);

    size_t feature_count = raw_data.size();
    thread_local std::vector<float> buffer;
    buffer.resize(feature_count);
    for (size_t i = 0; i < feature_count; ++i) {
        buffer[i] = static_cast<float>(raw_data[i]);
    }
    auto X_tensor = torch::from_blob(buffer.data(), { 1, static_cast<long>(feature_count) }, torch::kFloat32).clone();
    auto X_norm = (X_tensor - X_mean) / X_scale;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(X_norm);

    int old_threads = torch::get_num_threads();
    torch::set_num_threads(1);
    torch::NoGradGuard no_grad;
    auto preds_norm = module.forward(inputs).toTensor();
    torch::set_num_threads(old_threads);

    auto preds_orig = preds_norm * Y_scale + Y_mean;
    return static_cast<double>(preds_orig.item<float>()) * v0_norm;
}

// Overload for raw state data: map with nn_input_1_rotate_lambert then call fast_predict_vector
inline std::vector<double> fast_predict_vector_raw_data(
    const std::vector<std::vector<double>>& raw_data,
    const torch::Tensor& X_mean,
    const torch::Tensor& X_scale,
    const torch::Tensor& Y_mean,
    const torch::Tensor& Y_scale,
    torch::jit::script::Module& module,
    int type)
{
    std::vector<std::vector<double>> map_data;
    map_data.reserve(raw_data.size());
    for (const auto& row : raw_data) {
        map_data.push_back(nn_input_1_rotate_lambert(row));
    }
    return fast_predict_vector(map_data, X_mean, X_scale, Y_mean, Y_scale, module, type);
}

inline double fast_predict_vector_raw_data(
    const std::vector<double>& raw_data,
    const torch::Tensor& X_mean,
    const torch::Tensor& X_scale,
    const torch::Tensor& Y_mean,
    const torch::Tensor& Y_scale,
    torch::jit::script::Module& module,
    int type)
{
    auto map_data = nn_input_1_rotate_lambert(raw_data);
    return fast_predict_vector(map_data, X_mean, X_scale, Y_mean, Y_scale, module, type);
}

inline void libtorch_model_read(std::string libtorch_model_path, torch::Tensor& X_mean_tensor, torch::Tensor& X_scale_tensor, torch::Tensor& Y_mean_tensor, torch::Tensor& Y_scale_tensor, torch::jit::script::Module& module)
{
    // -------------------------------
    // Normalize input data: load scaler_X.csv (expects 17 rows of mean and scale)
    // -------------------------------
    std::string scalerX_csv = libtorch_model_path + "/scaler_X.csv";
    auto scalerX = read_scaler_csv(scalerX_csv);
    std::vector<double> X_means = scalerX.first;
    std::vector<double> X_scales = scalerX.second;
    int final_cols = X_scales.size();

    if (X_means.size() != final_cols || X_scales.size() != final_cols) {
        std::cerr << "Number of scaler_X parameters (" << X_means.size()
            << ") does not match number of input features (" << final_cols << ")!" << std::endl;
        return;
    }
    X_mean_tensor = torch::from_blob(X_means.data(), { (long)final_cols }, torch::kFloat64)
        .to(torch::kFloat32)
        .unsqueeze(0)
        .clone();
    X_scale_tensor = torch::from_blob(X_scales.data(), { (long)final_cols }, torch::kFloat64)
        .to(torch::kFloat32)
        .unsqueeze(0)
        .clone();

    // Load Y scaler
    std::string scalerY_csv = libtorch_model_path + "/scaler_Y.csv";
    auto scalerY = read_scaler_csv(scalerY_csv);
    std::vector<double> Y_means = scalerY.first;
    std::vector<double> Y_scales = scalerY.second;
    if (Y_means.size() != 1 || Y_scales.size() != 1) {
        std::cerr << "Scaler_Y parameter count is not 1!" << std::endl;
        return;
    }
    Y_mean_tensor = torch::tensor(Y_means[0], torch::kFloat32);
    Y_scale_tensor = torch::tensor(Y_scales[0], torch::kFloat32);

    // -------------------------------
    // Load TorchScript model (17-dim input, 1-dim output)
    // -------------------------------
    std::string model_file = libtorch_model_path + "/model_cpu.pt";
    try {
        module = torch::jit::load(model_file);
    }
    catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << model_file << std::endl;
        return;
    }
    module.eval();
    std::cout << "Model loaded successfully!" << std::endl;

    torch::NoGradGuard no_grad;
    return;
}
