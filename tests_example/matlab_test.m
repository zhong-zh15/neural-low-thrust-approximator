% test_nn_lowthrust.m
clc
clear all
close all

%% input
rv0 = [153200115041.47125; -371861548991.51477; -2457827991.595745;
       16946.19084502839;    7728.20307500515;  384.4824219631707];
rvt = [388897087868.8704;  -26556461186.848797;  -6811565802.344083;
       1823.093901430057;  18678.115133813287;  -865.3249902778101];
dt   = 23112000.0;
m0   = 2500.0;
Tmax = 0.3;
Isp  = 3000.0;
mu   = 1.32712440018e20;

inputVec = [rv0; rvt; dt; m0; Tmax; Isp; mu];

%% 2. process
processed = nn_input_1_rotate_lambert(inputVec);

%% 3. delta-v 
modelDirDV = fullfile(pwd, '../models', 'eigen_model_large');
dv = fast_predict_vector(modelDirDV, inputVec);
fprintf('Optimal Control result: 2019.66 m/s, EIGEN predicted first result: %.2f m/s\n', dv);

%% 4. tmin 
modelDirTmin = fullfile(pwd, '../models', 'eigen_model_tmin_large');
tmin = fast_predict_vector_tmin(modelDirTmin, inputVec);
fprintf('Optimal Control result: 2.0781e+07 s, EIGEN predicted first result: %.2f s\n', tmin);
