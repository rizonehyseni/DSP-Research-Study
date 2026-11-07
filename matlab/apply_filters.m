% APPLY_FILTERS  Makes the filtered versions of all 450 clips.
%
% Run from the repository folder (DSP-Research-Study), after setting the two
% lines below. Reads   <driveFolder>/data/raw/<source>/<clip>.wav
% and writes           <driveFolder>/data/filtered/<clip>__<filter>_<strength>.wav
% Files that already exist are skipped, so you can stop and run it again.

driveFolder = 'G:\My Drive\dsp-research-study';   % change to your Drive path
myFilter    = 'quantize';                          % your filter: mp3, opus, lowpass, resample, quantize, median

% Strengths of every filter (do not change: everyone uses the same values)
if strcmp(myFilter, 'mp3'),      strengths = [64 32 16];        end   % kbps
if strcmp(myFilter, 'opus'),     strengths = [24 12 6];         end   % kbps
if strcmp(myFilter, 'lowpass'),  strengths = [7000 5000 3000];  end   % cutoff, Hz
if strcmp(myFilter, 'resample'), strengths = [12000 8000 6000]; end   % middle rate, Hz
if strcmp(myFilter, 'quantize'), strengths = [12 8 6];          end   % bits
if strcmp(myFilter, 'median'),   strengths = [3 5 7];           end   % window, samples

addpath('matlab/filters', 'matlab/utils');
outFolder = fullfile(driveFolder, 'data', 'filtered');
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

% 1. Find all original clips (one folder per source: SSJ, AMSE, ..., LibriSpeech)
sources = {'SSJ', 'AMSE', 'BoN', 'AdvWave', 'AHarm', 'LibriSpeech'};
files = {};
for k = 1:length(sources)
    list = dir(fullfile(driveFolder, 'data', 'raw', sources{k}, '*.wav'));
    for j = 1:length(list)
        files{end+1} = fullfile(list(j).folder, list(j).name); %#ok<SAGROW>
    end
end
fprintf('%d clips to filter with %s\n', length(files), myFilter);

% 2. Filter every clip with the three strengths and save the result
results = {};
for i = 1:length(files)
    [x, fs] = audioread(files{i});
    x = mean(x, 2);                                % mono
    [~, clip] = fileparts(files{i});
    for s = strengths
        outFile = fullfile(outFolder, sprintf('%s__%s_%d.wav', clip, myFilter, s));
        if exist(outFile, 'file')
            y = audioread(outFile);                % already done earlier
        else
            y = feval(['filter_' myFilter], x, fs, s);
            audiowrite(outFile, y, fs);
        end
        snr = 10*log10(sum(x.^2) / max(sum((x - y).^2), eps));
        results(end+1, :) = {clip, myFilter, s, snr}; %#ok<SAGROW>
    end
    fprintf('%d/%d %s done\n', i, length(files), clip);
end

% 3. Save how much each clip changed (SNR in dB, higher = smaller change)
T = cell2table(results, 'VariableNames', {'clip', 'filter', 'strength', 'snr_db'});
writetable(T, fullfile(outFolder, ['snr_log_' myFilter '.csv']));
fprintf('Done: %d filtered files\n', height(T));
