"""Download the clips for the main experiment and write the manifest.

  - JALMBench: N clips per subset (SSJ, AMSE, BoN, AdvWave, AHarm), one attempt per question
  - LibriSpeech test-clean: M clips of normal speech

Audio is saved as 16 kHz mono WAV in <drive>/data/raw/<source>/<clip_id>.wav
and listed in <drive>/data/manifest.csv.

Run in Colab:
    !python scripts/prepare_data.py --drive "/content/drive/MyDrive/dsp-research-study"
"""

import argparse
import csv
import os
from collections import Counter

import pyarrow.parquet as pq
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from check_setup import ATTACKS, DEFAULT_DRIVE, JALMBENCH_ID, SR, decode, raw_audio

FIELDS = ["clip_id", "source", "path", "text", "original_text", "query_id",
          "attempt_id", "target_model", "edit", "duration_s"]


def stream_subset(dataset_id, subset):
    """Yield rows of one JALMBench subset; fall back to the parquet shards if the config is empty."""
    found = False
    for row in raw_audio(load_dataset(dataset_id, subset, split="train", streaming=True)):
        found = True
        yield row
    if found:
        return
    shards = sorted(f for f in list_repo_files(dataset_id, repo_type="dataset")
                    if f"/{subset}/" in f and f.endswith(".parquet"))
    for shard in shards:
        path = hf_hub_download(dataset_id, shard, repo_type="dataset")
        yield from pq.read_table(path).to_pylist()


def save(clip_id, source, audio, out_root):
    folder = os.path.join(out_root, source)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, clip_id + ".wav")
    sf.write(path, audio, SR)
    return path


def jalmbench_clips(dataset_id, subset, n, out_root):
    """First n distinct questions of a subset (the first attempt of each)."""
    seen, rows = set(), []
    for row in stream_subset(dataset_id, subset):
        query_id = str(row.get("id", len(seen)))
        if query_id in seen:
            continue
        seen.add(query_id)
        audio = decode(row["audio"])
        clip_id = f"{subset}_{len(rows):03d}"
        rows.append({
            "clip_id": clip_id, "source": subset,
            "path": save(clip_id, subset, audio, out_root),
            "text": row.get("text") or row.get("prompt") or "",
            "original_text": row.get("original_text", ""),
            "query_id": query_id,
            "attempt_id": row.get("attempt_id", ""),
            "target_model": row.get("target_model", ""),
            "edit": row.get("edit", ""),
            "duration_s": round(len(audio) / SR, 2),
        })
        if len(rows) == n:
            break
    targets = Counter(r["target_model"] for r in rows if r["target_model"])
    print(f"  {subset}: {len(rows)} clips" + (f", target_model: {dict(targets)}" if targets else ""))
    return rows


def librispeech_clips(m, out_root):
    ds = raw_audio(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True))
    rows = []
    for row in ds:
        audio = decode(row["audio"])
        clip_id = f"LibriSpeech_{len(rows):03d}"
        rows.append({
            "clip_id": clip_id, "source": "LibriSpeech",
            "path": save(clip_id, "LibriSpeech", audio, out_root),
            "text": row["text"], "original_text": "", "query_id": row.get("id", ""),
            "attempt_id": "", "target_model": "", "edit": "",
            "duration_s": round(len(audio) / SR, 2),
        })
        if len(rows) == m:
            break
    print(f"  LibriSpeech: {len(rows)} clips")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", default=DEFAULT_DRIVE, help="shared Drive folder")
    parser.add_argument("--jalmbench", default=JALMBENCH_ID)
    parser.add_argument("--n", type=int, default=50, help="clips per JALMBench subset")
    parser.add_argument("--m", type=int, default=200, help="LibriSpeech clips")
    args = parser.parse_args()

    out_root = os.path.join(args.drive, "data", "raw")
    rows = []
    for subset in ATTACKS:
        rows += jalmbench_clips(args.jalmbench, subset, args.n, out_root)
    rows += librispeech_clips(args.m, out_root)

    manifest = os.path.join(args.drive, "data", "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone: {len(rows)} clips. Manifest: {manifest}")


if __name__ == "__main__":
    main()
