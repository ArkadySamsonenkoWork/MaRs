clear;
base = fileparts(mfilename('fullpath'));
output_folder = fullfile(base, 'easyspin_spectra');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end
names = {'isc_triplet', 'recombination_triplet', 'radical_pair'};
for i = 1:numel(names)
    Sys = struct();
    Exp = struct();
    Sys.lw = [1.0 1.0];
    Exp.Range = [300 400];
    Exp.mwFreq = 9.8;
    Exp.Harmonic = 0;
    Exp.nPoints = 1024;
    if i <= 2
        Sys.S = 1;
        Sys.g = [2.02 2.02 2.02];
        Sys.D = [500 100];
        if i == 1
            Sys.HStrain = 1;
            Sys.initState = {[0.2 0.3 0.5], 'xyz'};
        else
            Sys.initState = {[0 1 0], 'eigen'};
        end
    else
        Sys.S = [1/2 1/2];
        Sys.g = [2.02 2.03 2.04; 2.03 2.025 2.025];
        Sys.J = 400;
        Sys.dip = [-23.3333333333 -43.3333333333 66.6666666667];
        singlet = [0; 1; -1; 0]/sqrt(2);
        Sys.initState = {singlet*singlet', 'uncoupled'};
    end
    [field, spectrum] = pepper(Sys, Exp);
    writematrix([field(:)/1000, spectrum(:)], fullfile(output_folder, [names{i} '.csv']));
    figure;
    plot(field/1000, spectrum);
    xlabel('Field (T)');
    title(names{i}, 'Interpreter', 'none');
end
