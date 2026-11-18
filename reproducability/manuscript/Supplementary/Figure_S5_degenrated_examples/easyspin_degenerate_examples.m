root = fileparts(mfilename('fullpath'));
Opt = struct('GridSize', [80 0], 'ModellingAccuracy', 1e-7, ...
             'Threshold', 1e-4, 'Method', 'matrix');

for k = 1:4
    data = load(fullfile(root, sprintf('example_%d.mat', k)));
    Sys = data.Sys;
    Exp = data.Exp;
    Exp.Harmonic = 1;
    [B, y] = pepper(Sys, Exp, Opt);
    writematrix([B(:), y(:)], fullfile(root, sprintf('example_%d_easyspin.csv', k)));
    figure('Name', sprintf('Example %d', k));
    plot(B, y, 'k-', 'LineWidth', 1);
    xlabel('B (mT)', 'FontSize', 10);
    set(gca, 'FontSize', 10, 'Box', 'off', 'YColor', 'none');
end
