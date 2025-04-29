#include "mex.h"
#include "eigen_nn.h"    // Declaration of nn_input_1_rotate_lambert

// Entry point for MEX
void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("nn_lowthrust:nn_input_1_rotate_lambert:InvalidNumInputs",
            "Exactly one input vector is required.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("nn_lowthrust:nn_input_1_rotate_lambert:InvalidNumOutputs",
            "Only one output is allowed.");
    }

    // Convert input mxArray to std::vector<double>
    double* inputPtr = mxGetPr(prhs[0]);
    mwSize numElems = mxGetNumberOfElements(prhs[0]);
    std::vector<double> rawInput(inputPtr, inputPtr + numElems);

    // Call the C++ preprocessing function
    std::vector<double> processed = nn_input_1_rotate_lambert(rawInput);

    // Create output MATLAB column vector
    mwSize outLen = processed.size();
    plhs[0] = mxCreateDoubleMatrix(outLen, 1, mxREAL);
    double* outputPtr = mxGetPr(plhs[0]);
    for (mwSize i = 0; i < outLen; ++i) {
        outputPtr[i] = processed[i];
    }
}
