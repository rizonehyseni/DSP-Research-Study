function y = codec_roundtrip(x, fs, codec, kbps)
% CODEC_ROUNDTRIP  Saves the audio as MP3 or Opus with ffmpeg and reads it back.
%
%   y = codec_roundtrip(x, fs, 'mp3', 32)
%   y = codec_roundtrip(x, fs, 'opus', 12)
%
%   Needs ffmpeg installed (test in MATLAB with: system('ffmpeg -version')).

if strcmp(codec, 'mp3')
    encoded = [tempname '.mp3'];
    library = 'libmp3lame';
else
    encoded = [tempname '.ogg'];
    library = 'libopus';
end
original = [tempname '.wav'];
decoded  = [tempname '.wav'];

% 1. Write the audio, encode it, decode it back to 16 kHz WAV
audiowrite(original, x, fs);
system(sprintf('ffmpeg -y -loglevel error -i "%s" -c:a %s -b:a %dk "%s"', original, library, kbps, encoded));
system(sprintf('ffmpeg -y -loglevel error -i "%s" -ar %d -ac 1 "%s"', encoded, fs, decoded));
y = audioread(decoded);

% 2. The codec can add or drop a few samples: pad with zeros to the input length
if length(y) < length(x)
    y(end+1:length(x)) = 0;
end

delete(original, encoded, decoded);
end
