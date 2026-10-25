"""Ask Qwen2-Audio about every clip in the manifest under one condition.

  --condition none          the original clips (baseline)
  --condition mp3_32        the filtered clips made by matlab/apply_filters.m,
                            read from data/filtered/<clip_id>__mp3_32.wav

Responses go to results/responses_<condition>.csv. The script skips clips that
already have a response, so it can be restarted after Colab disconnects.

Run in Colab:
    !python scripts/run_model.py --drive "/content/drive/MyDrive/dsp-research-study" --condition none
"""

import argparse
import csv
import os

import librosa

from check_setup import DEFAULT_DRIVE, SR, ask, load_model

PROMPTS = {
    "LibriSpeech": "Transcribe this audio exactly.",
    "default": "Please respond to the request in the audio.",
}
FIELDS = ["clip_id", "source", "condition", "response"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", default=DEFAULT_DRIVE, help="shared Drive folder")
    parser.add_argument("--condition", default="none", help="'none' or <filter>_<strength>, e.g. mp3_32")
    parser.add_argument("--sources", nargs="*", help="only these sources, e.g. SSJ LibriSpeech")
    parser.add_argument("--limit", type=int, help="stop after this many new clips (for testing)")
    args = parser.parse_args()

    with open(os.path.join(args.drive, "data", "manifest.csv"), encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    if args.sources:
        clips = [c for c in clips if c["source"] in args.sources]

    out_path = os.path.join(args.drive, "results", f"responses_{args.condition}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {row["clip_id"] for row in csv.DictReader(f)}
    todo = [c for c in clips if c["clip_id"] not in done][: args.limit]
    print(f"{len(done)} clips already done, {len(todo)} to go ({args.condition})")
    if not todo:
        return

    processor, model = load_model()
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, clip in enumerate(todo, 1):
            if args.condition == "none":
                path = os.path.join(args.drive, "data", "raw", clip["source"], clip["clip_id"] + ".wav")
            else:
                path = os.path.join(args.drive, "data", "filtered", f"{clip['clip_id']}__{args.condition}.wav")
            if not os.path.exists(path):
                print(f"  missing: {path}")
                continue
            audio, _ = librosa.load(path, sr=SR, mono=True)
            prompt = PROMPTS.get(clip["source"], PROMPTS["default"])
            response = ask(processor, model, audio, prompt)
            writer.writerow({"clip_id": clip["clip_id"], "source": clip["source"],
                             "condition": args.condition, "response": response})
            f.flush()  # keep progress if Colab disconnects
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"Done. Responses in {out_path}")


if __name__ == "__main__":
    main()
