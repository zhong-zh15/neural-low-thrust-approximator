#pragma once

#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include "eigen_nn_csvreader.h"
#include "nn_input_map.h"

// Compute the minimum delta-v based on Lambert's solution
inline void calculate_lambert_dv(const std::vector<double>& raw_data, double& dv_lambert_min)
{
    double dv_upper = std::max(raw_data[4], raw_data[5]);
    double dv_lower = std::min(raw_data[4], raw_data[5]);
    // Enforce a minimum lower bound of 10.0
    dv_lower = dv_lower < 10.0 ? 10.0 : dv_lower;
    double ratio = dv_lower / dv_upper;
    // Apply scaling factor if ratio is very small
    if (ratio < 0.02)
        ratio *= 50;
    else
        ratio = 1.0;
    double dv_max_temp = dv_upper / ratio;
    // Ensure at least 10% of the sum of both values
    dv_lambert_min = std::max((raw_data[4] + raw_data[5]) * 0.10, dv_max_temp);
}

//------------------------------------------------------------------------------
// EigenFastPredictor_small
//
// Performs forward propagation using Eigen with parameters stored as floats.
// Loads model data from CSV files:
//   - Input normalization: X_mean.csv, X_scale.csv
//   - Output normalization: Y_mean.csv, Y_scale.csv
//   - Layer parameters: layerN_weight.csv (matrix), layerN_bias.csv (vector)
//------------------------------------------------------------------------------
class EigenFastPredictor_small {
public:
    Eigen::VectorXf X_mean;   // Input means
    Eigen::VectorXf X_scale;  // Input scales
    Eigen::VectorXf X_inv;    // Inverse of input scales
    float Y_mean;             // Output mean
    float Y_scale;            // Output scale

    // Layer parameters: weights and biases for each layer
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::VectorXf> biases;

    // Constructor: model_dir contains all CSV files.
    // num_layers sets how many linear layers to load (default 4).
    EigenFastPredictor_small(const std::string& model_dir, int num_layers = 4) {
        // Load and prepare input normalization
        X_mean = loadCSVVector(model_dir + "/X_mean.csv");
        X_scale = loadCSVVector(model_dir + "/X_scale.csv");
        X_inv = X_scale.cast<float>().array().inverse().matrix();

        // Load and validate output normalization
        Eigen::VectorXf ymeanVec = loadCSVVector(model_dir + "/Y_mean.csv");
        Eigen::VectorXf yscaleVec = loadCSVVector(model_dir + "/Y_scale.csv");
        if (ymeanVec.size() < 1 || yscaleVec.size() < 1) {
            throw std::runtime_error("Y_mean.csv or Y_scale.csv is empty");
        }
        Y_mean = ymeanVec(0);
        Y_scale = yscaleVec(0);

        // Load each layer's weights and biases
        weights.resize(num_layers);
        biases.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            std::stringstream ss_w, ss_b;
            ss_w << model_dir << "/layer" << i << "_weight.csv";
            ss_b << model_dir << "/layer" << i << "_bias.csv";
            weights[i] = loadCSVMatrix(ss_w.str());
            biases[i] = loadCSVVector(ss_b.str());
        }
    }

    // Predict using raw input vector (delta-v normalization)
    double fast_predict_vector(const std::vector<double>& raw_data) const {
        double norm_factor = 1.0;
        // Normalize input using external function
        std::vector<double> normalized = nn_input_2_normlization_dv(raw_data, norm_factor);

        // Thread-local buffers for Eigen operations
        thread_local Eigen::VectorXf buf_input(X_mean.size());
        thread_local Eigen::VectorXf buf_tmp(weights[0].rows());

        // Map normalized double vector to float Eigen vector
        Eigen::VectorXf raw = Eigen::Map<const Eigen::VectorXd>(normalized.data(), normalized.size()).cast<float>();
        // Apply normalization: (raw - mean) * inv_scale
        buf_input = (raw.array() - X_mean.array()) * X_inv.array();

        int layers = weights.size();
        // Hidden layers with ReLU activation
        for (int i = 0; i < layers - 1; ++i) {
            buf_tmp.noalias() = weights[i] * buf_input + biases[i];
            buf_input = buf_tmp.cwiseMax(0.f);
        }
        // Output layer (no activation)
        buf_tmp.noalias() = weights[layers - 1] * buf_input + biases[layers - 1];
        float y_norm = buf_tmp(0);
        double y = static_cast<double>(y_norm * Y_scale + Y_mean);
        y *= norm_factor;

        // Apply minimum based on Lambert's delta-v
        double dv_min;
        calculate_lambert_dv(raw_data, dv_min);
        return std::max(dv_min, y);
    }

    // Predict using raw input vector (time-min normalization)
    double fast_predict_vector_tmin(const std::vector<double>& raw_data) const {
        double norm_factor = 1.0;
        std::vector<double> normalized = nn_input_2_normlization_tmin(raw_data, norm_factor);

        thread_local Eigen::VectorXf buf_input(X_mean.size());
        thread_local Eigen::VectorXf buf_tmp(weights[0].rows());

        Eigen::VectorXf raw = Eigen::Map<const Eigen::VectorXd>(normalized.data(), normalized.size()).cast<float>();
        buf_input = (raw.array() - X_mean.array()) * X_inv.array();

        int layers = weights.size();
        for (int i = 0; i < layers - 1; ++i) {
            buf_tmp.noalias() = weights[i] * buf_input + biases[i];
            buf_input = buf_tmp.cwiseMax(0.f);
        }
        buf_tmp.noalias() = weights[layers - 1] * buf_input + biases[layers - 1];
        float y_norm = buf_tmp(0);
        double y = static_cast<double>(y_norm * Y_scale + Y_mean);
        y *= norm_factor;

        double dv_min;
        calculate_lambert_dv(raw_data, dv_min);

        double mass = raw_data[15];
        double a0 = 0.1 / mass;  // Compute base acceleration
        double tmin_lambert = dv_min / a0;
        return std::max(tmin_lambert, y);
    }
};

//------------------------------------------------------------------------------
// Fixed-size layer and network definitions for optimized predictor
// Adjust these constants to change input/output dimensions or network depth
static constexpr int INPUT_DIM = 16;
static constexpr int HIDDEN_DIM = 128;
static constexpr int OUTPUT_DIM = 1;
static constexpr int NUM_LAYERS = 10;

// Layer data structures with fixed sizes and row-major layout for alignment
struct Layer0Type {
    Eigen::Matrix<float, HIDDEN_DIM, INPUT_DIM, Eigen::RowMajor> weight;
    Eigen::Matrix<float, HIDDEN_DIM, 1> bias;
};

struct HiddenLayerType {
    Eigen::Matrix<float, HIDDEN_DIM, HIDDEN_DIM, Eigen::RowMajor> weight;
    Eigen::Matrix<float, HIDDEN_DIM, 1> bias;
};

struct OutputLayerType {
    Eigen::Matrix<float, OUTPUT_DIM, HIDDEN_DIM, Eigen::RowMajor> weight;
    Eigen::Matrix<float, OUTPUT_DIM, 1> bias;
};

// Inline function combining GEMV, bias addition, and ReLU activation for a hidden layer
template<typename LayerT>
inline void dense_relu(const LayerT& layer,
    const Eigen::Matrix<float, HIDDEN_DIM, 1>& in,
    Eigen::Matrix<float, HIDDEN_DIM, 1>& out) noexcept
{
    out.noalias() = layer.weight * in;   // GEMV: W * x
    out += layer.bias;                   // Add bias
    out = out.cwiseMax(0.f);             // ReLU activation
}

// Compile-time unrolled recursion over hidden layers for maximum performance
template<std::size_t IDX = 0>
inline void run_hidden(Eigen::Matrix<float, HIDDEN_DIM, 1>& h,
    Eigen::Matrix<float, HIDDEN_DIM, 1>& tmp,
    const std::array<HiddenLayerType, NUM_LAYERS - 2>& layers) noexcept
{
    if constexpr (IDX < NUM_LAYERS - 2) {
        dense_relu(layers[IDX], h, tmp);
        h.swap(tmp);
        run_hidden<IDX + 1>(h, tmp, layers);
    }
}

class EigenFastPredictor {
public:
    // Normalization parameters stored in fixed-size vectors for alignment
    Eigen::Matrix<float, INPUT_DIM, 1> X_mean;
    Eigen::Matrix<float, INPUT_DIM, 1> X_scale;
    Eigen::Matrix<float, INPUT_DIM, 1> X_inv;  // Inverse scales
    float Y_mean;
    float Y_scale;

    // Network layers stored as fixed-size structs for cache efficiency
    Layer0Type layer0;
    std::array<HiddenLayerType, NUM_LAYERS - 2> hiddenLayers;
    OutputLayerType layerOut;

    // Ensure aligned memory allocation for Eigen types
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructor: load all normalization and layer parameters from CSV files
        EigenFastPredictor(const std::string& model_dir) {
        // Load input normalization
            {
                Eigen::VectorXf temp = loadCSVVector(model_dir + "/X_mean.csv");
                if (temp.size() != INPUT_DIM)
                    throw std::runtime_error("X_mean.csv has incorrect dimension");
                X_mean = temp;

                temp = loadCSVVector(model_dir + "/X_scale.csv");
                if (temp.size() != INPUT_DIM)
                    throw std::runtime_error("X_scale.csv has incorrect dimension");
                X_scale = temp;
                X_inv = X_scale.array().inverse().matrix();
            }
            // Load output normalization
            {
                Eigen::VectorXf temp = loadCSVVector(model_dir + "/Y_mean.csv");
                if (temp.size() < 1)
                    throw std::runtime_error("Y_mean.csv is empty");
                Y_mean = temp(0);

                temp = loadCSVVector(model_dir + "/Y_scale.csv");
                if (temp.size() < 1)
                    throw std::runtime_error("Y_scale.csv is empty");
                Y_scale = temp(0);
            }

            // Load first layer parameters
            {
                std::stringstream ss_w, ss_b;
                ss_w << model_dir << "/layer0_weight.csv";
                ss_b << model_dir << "/layer0_bias.csv";
                Eigen::MatrixXf w = loadCSVMatrix(ss_w.str());
                Eigen::VectorXf b = loadCSVVector(ss_b.str());
                if (w.rows() != HIDDEN_DIM || w.cols() != INPUT_DIM)
                    throw std::runtime_error("layer0 weight has incorrect dimension");
                if (b.size() != HIDDEN_DIM)
                    throw std::runtime_error("layer0 bias has incorrect dimension");
                layer0.weight = w;
                layer0.bias = b;
            }
            // Load hidden layers
            for (int i = 1; i <= NUM_LAYERS - 2; ++i) {
                std::stringstream ss_w, ss_b;
                ss_w << model_dir << "/layer" << i << "_weight.csv";
                ss_b << model_dir << "/layer" << i << "_bias.csv";
                Eigen::MatrixXf w = loadCSVMatrix(ss_w.str());
                Eigen::VectorXf b = loadCSVVector(ss_b.str());
                if (w.rows() != HIDDEN_DIM || w.cols() != HIDDEN_DIM)
                    throw std::runtime_error("layer" + std::to_string(i) + " weight has incorrect dimension");
                if (b.size() != HIDDEN_DIM)
                    throw std::runtime_error("layer" + std::to_string(i) + " bias has incorrect dimension");
                hiddenLayers[i - 1].weight = w;
                hiddenLayers[i - 1].bias = b;
            }
            // Load output layer parameters
            {
                std::stringstream ss_w, ss_b;
                ss_w << model_dir << "/layer" << (NUM_LAYERS - 1) << "_weight.csv";
                ss_b << model_dir << "/layer" << (NUM_LAYERS - 1) << "_bias.csv";
                Eigen::MatrixXf w = loadCSVMatrix(ss_w.str());
                Eigen::VectorXf b = loadCSVVector(ss_b.str());
                if (w.rows() != OUTPUT_DIM || w.cols() != HIDDEN_DIM)
                    throw std::runtime_error("output layer weight has incorrect dimension");
                if (b.size() != OUTPUT_DIM)
                    throw std::runtime_error("output layer bias has incorrect dimension");
                layerOut.weight = w;
                layerOut.bias = b;
            }
    }

    // Fast prediction (delta-v normalization) using fixed-size network
    double fast_predict_vector(const std::vector<double>& raw_data) const {
        double norm_factor = 1.0;
        std::vector<double> normalized = nn_input_2_normlization_dv(raw_data, norm_factor);

        // Map input and normalize
        Eigen::Matrix<float, INPUT_DIM, 1> raw = Eigen::Map<const Eigen::VectorXd>(normalized.data(), normalized.size()).cast<float>();
        Eigen::Matrix<float, INPUT_DIM, 1> inp = (raw.array() - X_mean.array()) * X_inv.array();

        // First layer + ReLU
        Eigen::Matrix<float, HIDDEN_DIM, 1> h = layer0.weight * inp + layer0.bias;
        h = h.cwiseMax(0.f);

        // Hidden layers unrolled
        Eigen::Matrix<float, HIDDEN_DIM, 1> tmp;
        run_hidden(h, tmp, hiddenLayers);

        // Output layer
        Eigen::Matrix<float, OUTPUT_DIM, 1> out = layerOut.weight * h + layerOut.bias;
        float y_norm = out(0);
        double y = static_cast<double>(y_norm * Y_scale + Y_mean);
        y *= norm_factor;

        double dv_min;
        calculate_lambert_dv(raw_data, dv_min);
        return std::max(dv_min, y);
    }

    // Fast prediction (time-min normalization) using fixed-size network
    double fast_predict_vector_tmin(const std::vector<double>& raw_data) const {
        double norm_factor = 1.0;
        std::vector<double> normalized = nn_input_2_normlization_tmin(raw_data, norm_factor);

        Eigen::Matrix<float, INPUT_DIM, 1> raw = Eigen::Map<const Eigen::VectorXd>(normalized.data(), normalized.size()).cast<float>();
        Eigen::Matrix<float, INPUT_DIM, 1> inp = (raw.array() - X_mean.array()) * X_inv.array();

        Eigen::Matrix<float, HIDDEN_DIM, 1> h = layer0.weight * inp + layer0.bias;
        h = h.cwiseMax(0.f);

        Eigen::Matrix<float, HIDDEN_DIM, 1> tmp;
        run_hidden(h, tmp, hiddenLayers);

        Eigen::Matrix<float, OUTPUT_DIM, 1> out = layerOut.weight * h + layerOut.bias;
        float y_norm = out(0);
        double y = static_cast<double>(y_norm * Y_scale + Y_mean);
        y *= norm_factor;

        double dv_min;
        calculate_lambert_dv(raw_data, dv_min);
        double mass = raw_data[15];
        double a0 = 0.1 / mass;
        double tmin_lambert = dv_min / a0;
        return std::max(tmin_lambert, y);
    }
};
