mex -I"ann_x64\\include" -L"ann_x64" -l"ANN" -I"opencv210_x64\include" -L"opencv210_x64\lib" -l"cv210" -l"cxcore210" getConstraintsMatrix.cpp mexBase.cpp
mex -I"ann_x64\\include" -L"ann_x64" -l"ANN" -I"opencv210_x64\include" -L"opencv210_x64\lib" -l"cv210" -l"cxcore210" getContinuousConstraintMatrix.cpp mexBase.cpp
mex -I"ann_x64\\include" -L"ann_x64" -l"ANN" -I"opencv210_x64\include" -L"opencv210_x64\lib" -l"cv210" -l"cxcore210" getGridLLEMatrix.cpp mexBase.cpp LLE.cpp
mex -I"ann_x64\\include" -L"ann_x64" -l"ANN" -I"opencv210_x64\include" -L"opencv210_x64\lib" -l"cv210" -l"cxcore210" getGridLLEMatrixNormal.cpp mexBase.cpp LLE.cpp
mex -I"ann_x64\\include" -L"ann_x64" -l"ANN" -I"opencv210_x64\include" -L"opencv210_x64\lib" -l"cv210" -l"cxcore210" getNormalConstraintMatrix.cpp mexBase.cpp