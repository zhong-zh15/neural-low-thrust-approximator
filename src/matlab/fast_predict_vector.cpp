#include "mex.h"
#include "eigen_nn.h"    // Declarations of EigenFastPredictor and nn_input_1_rotate_lambert

// Entry point for MEX
void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
{
    // Expect model directory (string) and input vector
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nn_lowthrust:fast_predict_vector:InvalidNumInputs",
            "Two inputs required: model_dir (string) and input_vector.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("nn_lowthrust:fast_predict_vector:InvalidNumOutputs",
            "Only one output is allowed.");
    }

    // Convert first argument to std::string
    char* dirBuf = mxArrayToString(prhs[0]);
    std::string modelDir(dirBuf);
    mxFree(dirBuf);

    // Convert second argument to std::vector<double>
    double* inPtr = mxGetPr(prhs[1]);
    mwSize inLen = mxGetNumberOfElements(prhs[1]);
    std::vector<double> rawInput(inPtr, inPtr + inLen);

    // Preprocess inputs
    std::vector<double> processed = nn_input_1_rotate_lambert(rawInput);

    // Create predictor and run fast_predict_vector
    EigenFastPredictor predictor(modelDir);
    double deltaV = predictor.fast_predict_vector(processed);

    // Return scalar result
    plhs[0] = mxCreateDoubleScalar(deltaV);
}
