#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

// Utility functions for loading CSV data into Eigen matrices and vectors

// Split a string by a given delimiter and return a vector of tokens
template<typename T = std::string>
static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Read a CSV file and parse each row into a vector of floats
static std::vector<std::vector<float>> readCSV(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::vector<std::string> tokens = split(line, ',');
        std::vector<float> row;
        for (const std::string& tok : tokens) {
            std::istringstream iss(tok);
            float value;
            if (!(iss >> value)) {
                throw std::runtime_error("Failed to convert to float in file: " + filename);
            }
            row.push_back(value);
        }
        data.push_back(row);
    }
    infile.close();
    return data;
}

// Read a CSV file and parse each row into a vector of doubles
static std::vector<std::vector<double>> readCSV_d(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::vector<std::string> tokens = split(line, ',');
        std::vector<double> row;
        for (const std::string& tok : tokens) {
            std::istringstream iss(tok);
            double value;
            if (!(iss >> value)) {
                throw std::runtime_error("Failed to convert to double in file: " + filename);
            }
            row.push_back(value);
        }
        data.push_back(row);
    }
    infile.close();
    return data;
}

// Load a CSV file into an Eigen::MatrixXf
static Eigen::MatrixXf loadCSVMatrix(const std::string& filename) {
    auto data = readCSV(filename);
    if (data.empty()) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }
    int rows = static_cast<int>(data.size());
    int cols = static_cast<int>(data[0].size());
    Eigen::MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (static_cast<int>(data[i].size()) != cols) {
            throw std::runtime_error("Inconsistent column count in CSV file: " + filename);
        }
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}

// Load a CSV file into an Eigen::MatrixXd
static Eigen::MatrixXd loadCSVMatrix_d(const std::string& filename) {
    auto data = readCSV_d(filename);
    if (data.empty()) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }
    int rows = static_cast<int>(data.size());
    int cols = static_cast<int>(data[0].size());
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (static_cast<int>(data[i].size()) != cols) {
            throw std::runtime_error("Inconsistent column count in CSV file: " + filename);
        }
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}

// Load a CSV file into an Eigen vector (VectorXf or VectorXd)
// Accepts files with a single row or single column
static Eigen::VectorXf loadCSVVector(const std::string& filename) {
    Eigen::MatrixXf mat = loadCSVMatrix(filename);
    if (mat.rows() == 1) {
        return mat.transpose();
    }
    else if (mat.cols() == 1) {
        return mat;
    }
    else {
        throw std::runtime_error("CSV file could not be parsed as vector: " + filename);
    }
}

static Eigen::VectorXd loadCSVVector_d(const std::string& filename) {
    Eigen::MatrixXd mat = loadCSVMatrix_d(filename);
    if (mat.rows() == 1) {
        return mat.transpose();
    }
    else if (mat.cols() == 1) {
        return mat;
    }
    else {
        throw std::runtime_error("CSV file could not be parsed as vector: " + filename);
    }
}
