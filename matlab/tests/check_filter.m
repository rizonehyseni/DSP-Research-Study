function check_filter(name, strengths)
% CHECK_FILTER  Tests filter_<name> on a short test sound before using it on all clips.
%
%   check_filter('lowpass', [7000 5000 3000])
%
%   Run from the repository folder (DSP-Research-Study). Needs no data.

addpath('matlab/filters', 'matlab/utils');

% 1. Test sound: 1 second, a 440 Hz tone plus a quiet 6500 Hz tone
fs = 16000;
t = (0:fs-1)' / fs;
x = 0.5*sin(2*pi*440*t) + 0.1*sin(2*pi*6500*t);

% 2. Run the filter with each strength and check the result
for s = strengths
    y = feval(['filter_' name], x, fs, s);
    assert(length(y) == length(x), 'Output must have the same length as the input.');
    assert(size(y, 2) == 1, 'Output must be a column.');
    assert(~any(isnan(y)), 'Output contains NaN.');
    assert(max(abs(y)) <= 1, 'Output must stay between -1 and 1.');
    assert(~isequal(y, x), 'Output is the same as the input: the filter did nothing.');
    snr = 10*log10(sum(x.^2) / sum((x - y).^2));
    fprintf('filter_%s(%g): OK, SNR = %.1f dB\n', name, s, snr);
end
fprintf('All checks passed.\n');
end
