# dsp-jailbreak-study

_One-paragraph summary of the study goes here._

## Repository layout

```
dsp-jailbreak-study/
├── data/        # clip lists (audio files stay in Drive)
├── src/         # code: filters, model, evaluation
├── notebooks/   # Colab experiments
├── results/     # results as CSV
├── figures/     # plots for the paper
├── docs/        # notes, decisions, experiment log
└── paper/       # the paper
```

## Data

Audio files are **not** stored in this repository. They live in Google Drive;
`data/` only holds the clip lists that reference them. The `.gitignore` blocks
common audio formats so they can't be committed by accident.

## License

See [LICENSE](LICENSE).
