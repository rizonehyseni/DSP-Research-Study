function y = filter_quantize(x, fs, strength)
% FILTER_QUANTIZE  Keeps fewer bits per sample (strength = bits: 12, 8, 6).

% 1. Round every sample to 2^(bits-1) steps
steps = 2^(strength - 1);
y = round(x * steps) / steps;

% 2. Keep these two lines
y = y(1:length(x));        % same length as the input
y = min(max(y, -1), 1);    % values between -1 and 1
end
