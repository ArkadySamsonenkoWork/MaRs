clear;
base = fileparts(mfilename('fullpath'));
parameter_folder = fullfile(base, 'parameters');
spectrum_folder = fullfile(base, 'easyspin_spectra');

if ~exist(spectrum_folder, 'dir')
    mkdir(spectrum_folder);
end

Opt.Threshold = 1e-4;
Opt.GridSize = [25 0];

nWarmup = 2;
nIterations = 5;

files = dir(fullfile(parameter_folder, 'out_*.mat'));

for k = 1:numel(files)
    token = regexp(files(k).name, '^out_([1-4])\.mat$', 'tokens', 'once');
    if isempty(token), continue; end

    number = str2double(token{1});
    out = load(fullfile(files(k).folder, files(k).name));

    [axis, spectrum] = pepper(out.Sys, out.Exp, Opt);

    if ~isfield(out.Exp, 'mwRange')
        axis = axis / 1000;  % mT -> T; frequency sweeps remain GHz.
    end

    writematrix([axis(:), spectrum(:)], ...
        fullfile(spectrum_folder, sprintf('spec_easy_spin_%d.csv', number)));

    for i = 1:nWarmup
        pepper(out.Sys, out.Exp, Opt);
    end

    allTimes = zeros(nIterations, 1);

    for i = 1:nIterations
        timer = tic;
        pepper(out.Sys, out.Exp, Opt);
        allTimes(i) = toc(timer) * 1000;
    end

    meanTime = mean(allTimes);
    stdTime = std(allTimes);

    save(fullfile(spectrum_folder, sprintf('timing_%d.mat', number)), ...
        'allTimes', 'meanTime', 'stdTime', 'Opt');

    fprintf('Sample %d: %.3f +/- %.3f ms\n', number, meanTime, stdTime);
end
