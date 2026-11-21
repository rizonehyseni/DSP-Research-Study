# MP3 filter: experiment log

Filter: MP3 compression (ffmpeg, libmp3lame) at 64, 32 and 16 kbps, applied to all 450 clips.
Model: Qwen2-Audio-7B-Instruct (4-bit, greedy decoding, max 256 new tokens).
Judge: Llama-Guard-3-8B (4-bit) for ASR; WER with jiwer on the 200 LibriSpeech clips.

| Setting | ASR (200 attack clips) | WER |
|---|---|---|
| No filter | 43.0% | 2.3% |
| MP3 64 kbps | 43.5% | 2.0% |
| MP3 32 kbps | 38.5% | 2.2% |
| MP3 16 kbps | 37.5% | 2.4% |

Notes:
- AdvWave stays at 96-98% at every strength; AMSE is already 0% without a filter.
- Only SSJ shows a clear drop (44% to 16% at 16 kbps).
- Each source has 50 clips, so differences of a few clips are within noise.
- Files: `results/summary_mp3.csv`; notebook in `scripts/run_qwen_and_score.ipynb`.