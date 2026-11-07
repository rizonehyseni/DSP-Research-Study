function y = filter_NAME(x, fs, strength)
% FILTER_NAME  What the filter does (strength = meaning and unit: a, b, c).

% 1. Your filter (replace this line)
y = x;

% 2. Keep these two lines
y = y(1:length(x));        % same length as the input
y = min(max(y, -1), 1);    % values between -1 and 1
end
