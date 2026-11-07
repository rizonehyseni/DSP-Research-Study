% ANALYZE_NAME  Report figures: what filter NAME does to the sound.
%
% Copy to analyze_<yourfilter>.m, change the three lines below, and run it
% from the repository folder (after apply_filters.m has run for your filter).
% Saves three figures in figures/:
%   <name>_spectrogram.png   one attack clip before and after, 3 strengths
%   <name>_spectrum.png      average spectrum before and after (which frequencies are removed)
%   <name>_snr.png           how much the filter changes the sound (mean SNR per strength)

driveFolder = 'G:\My Drive\dsp-research-study';   % change to your Drive path
name        = 'NAME';      % your filter, e.g. 'lowpass'
strengths   = [0 0 0];     % your three strengths, e.g. [7000 5000 3000]

addpath('matlab/filters', 'matlab/utils');
if ~exist('figures', 'dir')
    mkdir('figures');
end

% The same attack clip for everyone
[x, fs] = audioread(fullfile(driveFolder, 'data', 'raw', 'AdvWave', 'AdvWave_000.wav'));
x = mean(x, 2);

% 1. Spectrogram: original + three filtered versions side by side
figure('Position', [100 100 1400 320]);
subplot(1, 4, 1);
spectrogram(x, hann(512), 384, 512, fs, 'yaxis');
title('Original');
for k = 1:3
    y = feval(['filter_' name], x, fs, strengths(k));
    subplot(1, 4, k + 1);
    spectrogram(y, hann(512), 384, 512, fs, 'yaxis');
    title(sprintf('%s %g', name, strengths(k)));
end
saveas(gcf, ['figures/' name '_spectrogram.png']);

% 2. Average spectrum (Welch): original vs. the three strengths, in one plot
figure;
[p, f] = pwelch(x, hann(512), 256, 512, fs);
plot(f, 10*log10(p), 'k', 'LineWidth', 1.5);
hold on;
for k = 1:3
    y = feval(['filter_' name], x, fs, strengths(k));
    [p, f] = pwelch(y, hann(512), 256, 512, fs);
    plot(f, 10*log10(p));
end
hold off;
xlabel('Frequency (Hz)');
ylabel('Power (dB/Hz)');
legend(["Original", string(name) + " " + string(strengths)]);
title(['Average spectrum: ' name]);
saveas(gcf, ['figures/' name '_spectrum.png']);

% 3. Mean SNR per strength over all 450 clips (from apply_filters.m)
T = readtable(fullfile(driveFolder, 'data', 'filtered', ['snr_log_' name '.csv']));
meanSnr = zeros(1, 3);
for k = 1:3
    meanSnr(k) = mean(T.snr_db(T.strength == strengths(k)));
end
figure;
bar(meanSnr);
xticklabels(string(strengths));
xlabel('Strength');
ylabel('Mean SNR (dB), higher = smaller change');
title(['How much ' name ' changes the sound']);
saveas(gcf, ['figures/' name '_snr.png']);

fprintf('Saved figures/%s_spectrogram.png, %s_spectrum.png, %s_snr.png\n', name, name, name);
