#include <iostream>
#include <vector>
#include "eigen_nn.h"

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
	std::vector<double> input = {rv0[0], rv0[1], rv0[2], rv0[3], rv0[4], rv0[5],
		rvt[0], rvt[1], rvt[2], rvt[3], rvt[4], rvt[5],
		dt,  m0, Tmax, Isp, mu};

    EigenFastPredictor* predictor = new EigenFastPredictor("./src/eigen_nn_lowthrust/eigen_model_large");
    std::vector<double> map_dv = nn_input_1_rotate_lambert(input);
    double out_dv = predictor->fast_predict_vector(map_dv);
    std::cout << "Optimal Control result: 2019.66 m/s, EIGEN predicted first result: " << out_dv << " m/s"<< std::endl;
	delete predictor;

    EigenFastPredictor* predictor_tmin = new EigenFastPredictor("./src/eigen_nn_lowthrust/eigen_model_tmin_large");
    std::vector<double> map_tmin = nn_input_1_rotate_lambert(input);
    double out_tmin = predictor_tmin->fast_predict_vector_tmin(map_tmin);
    std::cout << "Optimal Control result: 2.0781e+07 s, EIGEN predicted first result: " << out_tmin << " m/s" << std::endl;
	delete predictor_tmin;

    return 0;
}
