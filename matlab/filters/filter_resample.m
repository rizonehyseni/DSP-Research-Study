function y = filter_resample(x, fs, strength)
% FILTER_RESAMPLE  Goes down to a lower sample rate and back (strength = middle rate in Hz: 12000, 8000, 6000).

% 1. Down to the middle rate, then back to the original rate
y = resample(x, strength, fs);
y = resample(y, fs, strength);
if length(y) < length(x)
    y(end+1:length(x)) = 0;    % pad if a few samples were lost
end

% 2. Keep these two lines
y = y(1:length(x));        % same length as the input
y = min(max(y, -1), 1);    % values between -1 and 1
end