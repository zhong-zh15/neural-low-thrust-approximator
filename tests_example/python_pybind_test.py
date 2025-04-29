# Example script that calls the C++/Eigen predictor through pybind11
import neural_lowthrust as lt
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Initial and target state vectors
    rv0 = np.array([153200115041.47125, -371861548991.51477,
                    -2457827991.595745, 16946.19084502839,
                    7728.20307500515, 384.4824219631707])
    rvt = np.array([388897087868.8704, -26556461186.848797,
                    -6811565802.344083, 1823.093901430057,
                    18678.115133813287, -865.3249902778101])

    raw = list(rv0) + list(rvt) + [
           23_112_000.0,   # dt (s)
           2500.0,         # initial mass (kg)
           0.3,            # Tmax (N)
           3000.0,         # Isp (s)
           1.32712440018e20  # mu (m³/s²)
    ]


    model_dir = Path(__file__).resolve().parents[2] / "models" / "eigen_model_large"
    pred = lt.EigenFastPredictor(str(model_dir))
    dv = pred.fast_predict_vector(lt.nn_input_1_rotate_lambert(raw))

    model_dir = Path(__file__).resolve().parents[2] / "models" / "eigen_model_tmin_large"
    mapped = lt.nn_input_1_rotate_lambert(str(model_dir))
    tmin = pred.fast_predict_vector_tmin(mapped)


    print(f"Predicted delta-v: {dv:.2f} m/s")
    print(f"Predicted tmin: {tmin:.2f} s")
