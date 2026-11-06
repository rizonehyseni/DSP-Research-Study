% apply_filters.m
% Applies the 6 DSP filters (3 strengths each) to every WAV under inDir
% (data/raw/<source>/*.wav, the check_setup folder is skipped) and writes the
% results to outDir as <clip>__<filter>_<strength>.wav.
% Files that already exist are skipped, so the script can be restarted.
% Also writes snr_log.csv with the SNR of each filtered version vs. the original.
%
% Clips are 16 kHz mono, so everything above 8 kHz is already gone:
% low-pass and resampling strengths are chosen below 8 kHz.
%
% MP3 and Opus need ffmpeg installed (desktop MATLAB). If ffmpeg is missing
% those two filters are skipped with a warning.

inDir  = 'data/raw';   % Drive folder dsp-research-study/data/raw
outDir = 'data/filtered';
if ~exist(outDir, 'dir'), mkdir(outDir); end

filters = {
    'mp3',       [64 32 16];      % kbps
    'opus',      [24 12 6];       % kbps
    'lowpass',   [7000 5000 3000];% cutoff, Hz
    'resample',  [12000 8000 6000];% intermediate rate, Hz
    'quantize',  [12 8 6];        % bits
    'median',    [3 5 7]          % window, samples
};

% Each team member runs only their own filter: put its name here,
% e.g. onlyFilters = {'mp3'}; leave empty {} to run all six.
onlyFilters = {};
if ~isempty(onlyFilters)
    filters = filters(ismember(filters(:, 1), onlyFilters), :);
end

hasFfmpeg = system('ffmpeg -version > /dev/null 2>&1') == 0 || ...
            system('ffmpeg -version > NUL 2>&1') == 0;
if ~hasFfmpeg
    warning('ffmpeg not found: MP3 and Opus will be skipped.');
end

files = dir(fullfile(inDir, '*', '*.wav'));
files = files(~contains({files.folder}, 'check_setup'));
fprintf('%d clips to filter\n', numel(files));
logRows = {};
for i = 1:numel(files)
    [x, fs] = audioread(fullfile(files(i).folder, files(i).name));
    x = mean(x, 2);                          % mono
    [~, clip] = fileparts(files(i).name);
    for f = 1:size(filters, 1)
        name = filters{f, 1};
        for s = filters{f, 2}
            if any(strcmp(name, {'mp3', 'opus'})) && ~hasFfmpeg
                continue
            end
            outName = sprintf('%s__%s_%d.wav', clip, name, s);
            if exist(fullfile(outDir, outName), 'file')
                y = audioread(fullfile(outDir, outName));
            else
                y = apply_one(x, fs, name, s);
                y = y(1:min(end, numel(x)));     % same length as the original
                y(end+1:numel(x)) = 0;
                y = max(min(y, 1), -1);          % avoid clipping errors when writing
                audiowrite(fullfile(outDir, outName), y, fs);
            end
            snrDb = 10 * log10(sum(x.^2) / max(sum((x - y).^2), eps));
            logRows(end+1, :) = {clip, name, s, snrDb}; %#ok<SAGROW>
        end
    end
    fprintf('%d/%d %s done\n', i, numel(files), clip);
end

T = cell2table(logRows, 'VariableNames', {'clip', 'filter', 'strength', 'snr_db'});
if isempty(onlyFilters)
    logName = 'snr_log.csv';
else
    logName = sprintf('snr_log_%s.csv', strjoin(onlyFilters, '_'));
end
writetable(T, fullfile(outDir, logName));
fprintf('Done: %d filtered files in %s\n', height(T), outDir);


function y = apply_one(x, fs, name, s)
    switch name
        case 'lowpass'
            [z, p, k] = butter(8, s / (fs / 2));
            [sos, g] = zp2sos(z, p, k);
            y = filtfilt(sos, g, x);
        case 'resample'
            y = resample(resample(x, s, fs), fs, s);
        case 'quantize'
            levels = 2^(s - 1);
            y = round(x * levels) / levels;
        case 'median'
            y = medfilt1(x, s);
        case {'mp3', 'opus'}
            y = codec_roundtrip(x, fs, name, s);
    end
end


function y = codec_roundtrip(x, fs, codec, kbps)
    tmpIn  = [tempname '.wav'];
    tmpOut = [tempname '.wav'];
    if strcmp(codec, 'mp3')
        tmpEnc = [tempname '.mp3']; lib = 'libmp3lame';
    else
        tmpEnc = [tempname '.ogg']; lib = 'libopus';
    end
    audiowrite(tmpIn, x, fs);
    system(sprintf('ffmpeg -y -loglevel error -i "%s" -c:a %s -b:a %dk "%s"', tmpIn, lib, kbps, tmpEnc));
    system(sprintf('ffmpeg -y -loglevel error -i "%s" -ar %d -ac 1 "%s"', tmpEnc, fs, tmpOut));
    y = audioread(tmpOut);
    delete(tmpIn, tmpEnc, tmpOut);
end
