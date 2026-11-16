function y = filter_opus(x, fs, strength)

y = codec_roundtrip(x, fs, 'opus', strength);

y = y(1:length(x));
y = min(max(y, -1), 1);
end
