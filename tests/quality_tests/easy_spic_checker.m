clear;
out = load('example_parameters.mat');
T = readtable('easy_spin_checked/spectra_7.csv');

field_python = T.Var1 * 1000;
spec_python = T.Var2;

Sys = out.Sys;
Exp = out.Exp;
spec_easy_spin = pepper(Sys, Exp);
spec_easy_spin = spec_easy_spin / max(spec_easy_spin);
spec_python = spec_python / max(spec_python);
plot(field_python, spec_easy_spin, field_python, spec_python);
legend(["EasySpin", "Python"])
        