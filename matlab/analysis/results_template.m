% RESULTS_NAME  Report figure and table: does filter NAME stop the attacks, and at what cost?
%
% Copy to results_<yourfilter>.m, change the three lines below, and run it from the
% repository folder after Rizone has run evaluate.py (results/summary.csv exists).
% Saves figures/<name>_results.png and prints a small table for the report.

driveFolder = 'G:\My Drive\dsp-research-study';   % change to your Drive path
name        = 'NAME';      % your filter, e.g. 'lowpass'
strengths   = [0 0 0];     % your three strengths, e.g. [7000 5000 3000]

T = readtable(fullfile(driveFolder, 'results', 'summary.csv'), 'TextType', 'string');
conditions = ["none", string(name) + "_" + string(strengths)];   % no filter + 3 strengths
attacks = ["AdvWave", "BoN", "SSJ", "AMSE", "AHarm"];

% 1. ASR for every attack and condition, WER for LibriSpeech
asr = NaN(length(attacks), length(conditions));
wer = NaN(1, length(conditions));
for c = 1:length(conditions)
    for a = 1:length(attacks)
        row = T.condition == conditions(c) & T.source == attacks(a) & T.metric == "ASR";
        if any(row)
            asr(a, c) = T.value(row);
        end
    end
    row = T.condition == conditions(c) & T.source == "LibriSpeech" & T.metric == "WER";
    if any(row)
        wer(c) = T.value(row);
    end
end

% 2. Figure: ASR per attack (left) and WER (right), from no filter to strongest
figure('Position', [100 100 1100 380]);
subplot(1, 2, 1);
plot(1:4, asr', '-o', 'LineWidth', 1.5);
xticks(1:4);
xticklabels(conditions);
ylabel('ASR (%)  lower = better');
legend(attacks, 'Location', 'best');
title(['Attacks that got through: ' name]);
subplot(1, 2, 2);
bar(wer);
xticks(1:4);
xticklabels(conditions);
ylabel('WER (%)  lower = better');
title('Errors on normal speech');
if ~exist('figures', 'dir')
    mkdir('figures');
end
saveas(gcf, ['figures/' name '_results.png']);

% 3. Table for the report
disp(array2table([asr; wer], 'RowNames', cellstr([attacks, "WER LibriSpeech"]), ...
    'VariableNames', cellstr(conditions)));
