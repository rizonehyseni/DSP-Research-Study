function y = filter_lowpass(x, fs, strength)
% FILTER_LOWPASS  Cuts frequencies above the cutoff (strength = cutoff in Hz: 7000, 5000, 3000).

% 1. Zero-phase Butterworth low-pass filter
[b, a] = butter(8, strength / (fs / 2));
y = filtfilt(b, a, x);

% 2. Keep these two lines
y = y(1:length(x));        % same length as the input
y = min(max(y, -1), 1);    % values between -1 and 1
end