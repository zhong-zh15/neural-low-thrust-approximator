clc
clear all
close all
% build_mex.m  
projectRoot = pwd;
srcDir      = fullfile(projectRoot, '../cpp/eigen_version');
eigenInc1   = fullfile(srcDir, '');
eigenInc2   = fullfile(eigenInc1, 'Eigen');
matlabDir   = fullfile(projectRoot, '');

% core cpp codes
cpps = dir(fullfile(srcDir, '**', '*.cpp'));
core = {};
for k = 1:numel(cpps)
    core{end+1} = fullfile(cpps(k).folder, cpps(k).name);
end

wrappers = { ...
  'nn_input_1_rotate_lambert.cpp', ...
  'fast_predict_vector.cpp', ...
  'fast_predict_vector_tmin.cpp' ...
};


if ispc
    % MSVC: COMPFLAGS  /std:c++17 /O2
    stdFlag = 'COMPFLAGS="$COMPFLAGS /std:c++17 /O2"';
else
    % GCC/Clang: CXXFLAGS
    stdFlag = 'CXXFLAGS="$CXXFLAGS -std=c++17 -O3 -march=native"';
end

for i = 1:numel(wrappers)
    wrapperPath = fullfile(matlabDir, wrappers{i});
    fprintf('Building %s ...\n', wrappers{i});
    
    inc1 = ['-I', srcDir];
    inc2 = ['-I', eigenInc1];
    inc3 = ['-I', eigenInc2];
    
    mex(stdFlag, inc1, inc2, inc3, core{:}, wrapperPath);
end

fprintf('All MEX built successfully.\n');
