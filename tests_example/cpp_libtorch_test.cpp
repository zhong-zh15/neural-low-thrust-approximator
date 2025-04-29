#include <torch/torch.h>
#include "libtorch_nn.h"
int main() {

    double rv0[6] = { 153200115041.471252441406250, -371861548991.514770507812500, -2457827991.595745086669922,
                  16946.190845028388139, 7728.203075005149913, 384.482421963170736 };
    double rvt[6] = { 388897087868.870422363281250, -26556461186.848796844482422, -6811565802.344083786010742,
                      1823.093901430057258, 18678.115133813287684, -865.324990277810116 };
    double dt = 23112000.000000000000000;
    double m0 = 2500.0;
    double Tmax = 0.3;
    double Isp = 3000.0;

    double mu = 1.32712440018e20;
    double g0 = 9.80665;

    std::vector<double> input = { rv0[0], rv0[1], rv0[2], rv0[3], rv0[4], rv0[5],
        rvt[0], rvt[1], rvt[2], rvt[3], rvt[4], rvt[5],
        dt,  m0, Tmax, Isp, mu };

    {
        std::string libtorch_model_path = "../src/libtorch_nn_lowthrust/libtorch_model_dv";
        torch::Tensor X_mean_tensor, X_scale_tensor, Y_mean_tensor, Y_scale_tensor;
        torch::jit::script::Module module;
        libtorch_model_read(libtorch_model_path, X_mean_tensor, X_scale_tensor, Y_mean_tensor, Y_scale_tensor, module);
        auto rv_predictions = fast_predict_vector_raw_data(input, X_mean_tensor, X_scale_tensor, Y_mean_tensor, Y_scale_tensor, module, 1);
        std::cout << "Optimal Control result: 2019.66 m/s, Libtorch predicted first result: " << rv_predictions << " m/s" << std::endl;
    }

    {
        std::string libtorch_model_path = "../src/libtorch_nn_lowthrust/libtorch_model_tmin";
        torch::Tensor X_mean_tensor, X_scale_tensor, Y_mean_tensor, Y_scale_tensor;
        torch::jit::script::Module module;
        libtorch_model_read(libtorch_model_path, X_mean_tensor, X_scale_tensor, Y_mean_tensor, Y_scale_tensor, module);
        auto rv_predictions = fast_predict_vector_raw_data(input, X_mean_tensor, X_scale_tensor, Y_mean_tensor, Y_scale_tensor, module, 2);
        std::cout << "Optimal Control result: 2.0781e+07 s, Libtorch predicted first result: " << rv_predictions << " m/s" << std::endl;
    }

    return 0;
}