clear;
base = fileparts(mfilename('fullpath'));
parameter_folder = fullfile(base, 'parameters');
spectrum_folder = fullfile(base, 'easyspin_spectra');

if ~exist(spectrum_folder, 'dir'), mkdir(spectrum_folder); end

files = dir(fullfile(parameter_folder, 'spectra_*.mat'));

caseNames = strings(numel(files), 1);
statuses = strings(numel(files), 1);
messages = strings(numel(files), 1);
outputs = strings(numel(files), 1);

for i = 1:numel(files)
    [~, name] = fileparts(files(i).name);
    destination = fullfile(spectrum_folder, [name '.csv']);
    caseNames(i) = string(name);

    if exist(destination, 'file'), delete(destination); end

    try
        out = load(fullfile(files(i).folder, files(i).name));

        [field, spectrum] = pepper(out.Sys, out.Exp);

        % EasySpin returns mT; the notebook uses T. Keep raw intensity.
        writematrix([field(:)/1000, spectrum(:)], destination);

        statuses(i) = "saved";
        outputs(i) = string(destination);
        fprintf('Saved %s\n', destination);

    catch err
        statuses(i) = "failed";
        messages(i) = string(getReport(err, 'extended', 'hyperlinks', 'off'));
        fprintf(2, 'Failed %s: %s\n', name, err.message);
    end
end

report = table(caseNames, statuses, outputs, messages, ...
    'VariableNames', {'Case', 'Status', 'Output', 'Message'});

report_file = fullfile(spectrum_folder, 'generation_report.csv');
writetable(report, report_file);

fprintf('Saved %d/%d cases. Report: %s\n', ...
    sum(statuses == "saved"), numel(files), report_file);

if any(statuses == "failed")
    warning('Some EasySpin cases failed. Inspect generation_report.csv before comparing.');
end
