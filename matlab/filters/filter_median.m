function y = filter_median(x, fs, strength)
% FILTER_MEDIAN  Median filter that smooths sudden spikes (strength = window in samples: 3, 5, 7).

% 1. Replace each sample by the median of its neighbours
y = medfilt1(x, strength);

% 2. Keep these two lines
y = y(1:length(x));        % same length as the input
y = min(max(y, -1), 1);    % values between -1 and 1
end