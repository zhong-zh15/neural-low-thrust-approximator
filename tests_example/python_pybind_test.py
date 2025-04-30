# Example script that calls the C++/Eigen predictor through pybind11
from neural_lowthrust import nn_lowthrust as lt
import numpy as np
from importlib.metadata import distribution

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────
    # Initial and target state vectors
    rv0 = np.array([153200115041.47125, -371861548991.51477,
                    -2457827991.595745, 16946.19084502839,
                    7728.20307500515, 384.4824219631707])
    rvt = np.array([388897087868.8704, -26556461186.848797,
                    -6811565802.344083, 1823.093901430057,
                    18678.115133813287, -865.3249902778101])

    raw = list(rv0) + list(rvt) + [
           23112000.0,   # dt (s)
           2500.0,         # initial mass (kg)
           0.3,            # Tmax (N)
           3000.0,         # Isp (s)
           1.32712440018e20  # mu (m³/s²)
    ]

    # delta-v
    model_dir = str(distribution("neural_lowthrust").locate_file("neural_lowthrust/models/eigen_model_large"))
    pred = lt.EigenFastPredictor(model_dir)
    dv = pred.fast_predict_vector(lt.nn_input_1_rotate_lambert(raw))
    # tmin
    tmin_dir = str(distribution("neural_lowthrust").locate_file("neural_lowthrust/models/eigen_model_tmin_large"))
    pred_t = lt.EigenFastPredictor(tmin_dir)
    tmin = pred_t.fast_predict_vector_tmin(lt.nn_input_1_rotate_lambert(raw))

    print(f"Predicted delta-v: {dv:.2f} m/s")
    print(f"Predicted tmin:   {tmin:.2f} s")