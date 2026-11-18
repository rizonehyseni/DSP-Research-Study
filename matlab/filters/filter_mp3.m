function y = filter_mp3(x, fs, strength)
% FILTER_MP3  MP3 compression and decompression (strength = bitrate in kbps: 64, 32, 16).

% 1. Compress with ffmpeg and decode back
y = codec_roundtrip(x, fs, 'mp3', strength);

% 2. Keep these two lines
y = y(1:length(x));        % same length as the input
y = min(max(y, -1), 1);    % values between -1 and 1
end