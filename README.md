## Is Signal-Level Defense Enough? A Comparative Evaluation of DSP Transformations Against Audio Jailbreaks on Audio-Language Models


Voice AI models can be tricked by hidden, nearly inaudible noise in audio (adversarial jailbreaks). We test whether simple DSP filters, such as MP3 compression, low-pass filtering and resampling, can remove these attacks without hurting the model's understanding of normal speech.

## Research question

Which DSP transformation best blocks adversarial audio jailbreaks on audio-language models while keeping performance on normal speech?

## Method

1. Run attack audio and normal speech through the model with no filter (baseline).
2. Apply each DSP filter at several strengths.
3. Run the filtered audio through the model again.
4. Compare attack success rate and accuracy against the baseline.

## Structure

```
data/        clip lists (audio stored in Google Drive)
src/         filters, model, evaluation code
scripts/     experiment scripts
results/     results as CSV
figures/     plots
docs/        notes and experiment log
paper/       the paper
```

## Tools

Python, an open-source audio-language model, Google Colab, a public jailbreak benchmark and LibriSpeech.
