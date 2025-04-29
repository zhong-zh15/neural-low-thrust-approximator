
from pathlib import Path
from ptorch_nn_lowthrust import fast_predict_vector_raw_data, ptorch_model_read
if __name__ == "__main__":

    # ---------------- input ---------------------------------------------------
    rv0 = [
        153200115041.471252441406250,
        -371861548991.514770507812500,
        -2457827991.595745086669922,
        16946.190845028388139,
        7728.203075005149913,
        384.482421963170736,
    ]
    rvt = [
        388897087868.870422363281250,
        -26556461186.848796844482422,
        -6811565802.344083786010742,
        1823.093901430057258,
        18678.115133813287684,
        -865.324990277810116,
    ]
    dt = 23112000.0
    m0 = 2500.0
    Tmax = 0.3
    Isp = 3000.0
    mu = 1.32712440018e20

    # 拼接成 17 维原始向量
    input_vec = rv0 + rvt + [dt, m0, Tmax, Isp, mu]

    # ---------------- 1. Δv prediction -------------------------------------------------
    dv_model_dir = Path("../src/libtorch_nn_lowthrust/libtorch_model_dv")
    X_mean, X_scale, Y_mean, Y_scale, module = ptorch_model_read(dv_model_dir)

    pred_v = fast_predict_vector_raw_data(
        [input_vec], X_mean, X_scale, Y_mean, Y_scale, module, mode=1
    )[0]
    print(
        f"Optimal Control result (Reference): 2019.66 m/s, "
        f"Libtorch predicted (Python): {pred_v:.3f} m/s"
    )

    # ---------------- 2. tmin prediction --------------------------------------------
    tmin_model_dir = Path("../src/libtorch_nn_lowthrust/libtorch_model_tmin")
    X_mean, X_scale, Y_mean, Y_scale, module = ptorch_model_read(tmin_model_dir)
    pred_t = fast_predict_vector_raw_data(
        [input_vec], X_mean, X_scale, Y_mean, Y_scale, module, mode=2
    )[0]
    print(
        f"Optimal Control result (Reference): 2.0781e+07 s, "
        f"Libtorch predicted (Python): {pred_t:.3f} s"
    )

