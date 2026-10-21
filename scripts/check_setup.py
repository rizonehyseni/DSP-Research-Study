"""First check before the real experiment.

Checks that:
  1. a GPU is available,
  2. the shared Google Drive folder is mounted,
  3. JALMBench and LibriSpeech audio can be downloaded,
  4. Qwen2-Audio loads (4-bit) and answers a few clips without any filter.

Run in Colab (after mounting Drive and setting the HF_TOKEN secret):
    !python scripts/check_setup.py --jalmbench <HF dataset id from the JALMBench page>
"""

import argparse
import csv
import io
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, get_dataset_config_names, load_dataset

SR = 16000  # Qwen2-Audio expects 16 kHz mono
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
ATTACKS = ["SSJ", "AMSE", "BoN", "AdvWave", "A_Harm"]
DEFAULT_DRIVE = "/content/drive/MyDrive/dsp-jailbreak-study"


def decode(audio_field):
    """Turn a raw (decode=False) audio field into a 16 kHz mono float array."""
    data = audio_field.get("bytes")
    if data is None:
        data = open(audio_field["path"], "rb").read()
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        wav, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1) if wav.shape[1] < wav.shape[0] else wav.mean(axis=0)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def take(ds, n, source=None):
    """Take the first n rows (optionally only rows whose `source` matches)."""
    rows = []
    for row in ds:
        if source and str(row.get("source", "")).lower() != source.lower():
            continue
        rows.append(row)
        if len(rows) == n:
            break
    return rows


def load_jalmbench(dataset_id, n, max_scan):
    configs = get_dataset_config_names(dataset_id)
    print(f"  JALMBench configs: {configs}")
    by_name = {c.lower(): c for c in configs}
    clips = []
    for attack in ATTACKS:
        if attack.lower() in by_name:
            ds = load_dataset(dataset_id, by_name[attack.lower()], split="train", streaming=True)
            rows = take(ds.cast_column("audio", Audio(decode=False)), n)
        else:
            ds = load_dataset(dataset_id, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False)).take(max_scan)
            rows = take(ds, n, source=attack)
        print(f"  {attack}: {len(rows)} clips")
        for i, row in enumerate(rows):
            clips.append({"clip_id": f"{attack}_{i}", "source": attack,
                          "text": row.get("text", ""), "audio": decode(row["audio"])})
    return clips


def load_librispeech(n):
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    rows = take(ds.cast_column("audio", Audio(decode=False)), n)
    print(f"  LibriSpeech: {len(rows)} clips")
    return [{"clip_id": f"librispeech_{i}", "source": "LibriSpeech",
             "text": row["text"], "audio": decode(row["audio"])} for i, row in enumerate(rows)]


def load_model():
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2AudioForConditionalGeneration)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_compute_dtype=torch.float16),
    )
    return processor, model


def ask(processor, model, audio, prompt, max_new_tokens=256):
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "clip.wav"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    try:
        inputs = processor(text=text, audio=[audio], sampling_rate=SR, return_tensors="pt", padding=True)
    except TypeError:  # older transformers versions call the argument `audios`
        inputs = processor(text=text, audios=[audio], sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jalmbench", required=True, help="Hugging Face dataset id of JALMBench")
    parser.add_argument("--drive", default=DEFAULT_DRIVE, help="shared Drive folder")
    parser.add_argument("--n", type=int, default=1, help="clips per attack")
    parser.add_argument("--max-scan", type=int, default=5000, help="rows to scan when filtering by source")
    args = parser.parse_args()

    print("1. GPU")
    if not torch.cuda.is_available():
        raise SystemExit("  No GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
    print(f"  {torch.cuda.get_device_name(0)}")

    print("2. Google Drive")
    if not os.path.isdir(args.drive):
        raise SystemExit(f"  Folder not found: {args.drive}. Mount Drive and add the shared folder as a shortcut.")
    print(f"  OK: {args.drive}")

    print("3. Data")
    clips = load_jalmbench(args.jalmbench, args.n, args.max_scan) + load_librispeech(3)
    sample_dir = os.path.join(args.drive, "data", "raw", "check_setup")
    os.makedirs(sample_dir, exist_ok=True)
    for clip in clips:
        sf.write(os.path.join(sample_dir, clip["clip_id"] + ".wav"), clip["audio"], SR)
    print(f"  Saved the clips for listening in {sample_dir}")

    print("4. Model (first download takes a few minutes)")
    processor, model = load_model()
    results = []
    for clip in clips:
        if clip["source"] == "LibriSpeech":
            prompt = "Transcribe this audio exactly."
        else:
            prompt = "Please respond to the request in the audio."
        response = ask(processor, model, clip["audio"], prompt)
        results.append({"clip_id": clip["clip_id"], "source": clip["source"],
                        "reference_text": clip["text"], "response": response})
        print(f"\n[{clip['clip_id']}]\n  {response[:300]}")

    out_path = os.path.join(args.drive, "results", "check_setup.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDone. Responses saved to {out_path}")


if __name__ == "__main__":
    main()
