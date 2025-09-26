"""
T4 Streaming ASR micro-prototype (single script, no CLI)

What this does
--------------
- Loads a NeMo RNNT Conformer model from a local .nemo file
- Streams 20–40 ms audio frames from a Common Voice eval manifest
- Emits partial hypotheses during streaming and measures latency
- Computes WER (jiwer), RTF p50/p95, first-partial latency, and peak GPU memory

Notes
-----
- This is a *micro* prototype for design exploration. For true low-latency you’d
  deploy with Triton + TensorRT (single instance, no dynamic batching) and a
  streaming RNNT pipeline; this script focuses on local prototyping.
- Partial hypotheses are produced by re-decoding the accumulated buffer. This is
  not as efficient as proper streaming decode but is sufficient for measuring
  end-to-end latencies on a T4.

Dependencies
------------
- python -m pip install nemo_toolkit[asr] torch torchaudio jiwer soundfile numpy tqdm

Run
---
- Just run this file (configs are below):  python t4_streaming_asr_prototype.py
"""
from __future__ import annotations
import json, os, time, math, io, tempfile, statistics, shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from jiwer import wer

# ======== CONFIG (edit here, no CLI) =========================================
MANIFEST_PATH = Path("data/asr_eval/common_voice_17_0/manifest_en.jsonl")
MODEL_PATH = Path("models/en/stt_en_conformer_transducer_large.nemo")

# Streaming + engine
SAMPLE_RATE = 16000  # model expect rate (will resample if needed)
CHUNK_MS = 20        # frame size in ms (adjust manually for ablations)
USE_FP16 = True      # set False to run in FP32

# Real-time simulation controls (sleep to mimic wall-clock)
SIMULATE_REALTIME = False   # set True to include time.sleep matching audio pace
PARTIAL_EVERY_N_CHUNKS = 2  # emit a partial every N chunks to limit overhead

# Evaluation subset controls
MAX_TOTAL_AUDIO_MINUTES = 15.0  # stop once we’ve reached ~15 minutes of audio
MAX_UTTS = None  # or set e.g. 200 to cap number of utterances

# Decoding
DECODING_STRATEGY = "greedy"  # RNNT greedy; leave as default

# Output paths
RUN_DIR = Path("runs/t4_streaming_prototype")
PER_UTT_CSV = RUN_DIR / "per_utt_metrics.csv"
SUMMARY_CSV = RUN_DIR / "summary.csv"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_DEVICE_INDEX = 0
# ============================================================================

# Utility: simple CSV writer
import csv

def _makedirs(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    _makedirs(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ======== NeMo model loader ==================================================
try:
    from nemo.collections.asr.models import EncDecRNNTBPEModel
except Exception as e:
    raise RuntimeError(
        "NeMo is required (pip install nemo_toolkit[asr]). Original error: %r" % e
    )


def load_model(model_path: Path, device: str = DEVICE, use_fp16: bool = True):
    model = EncDecRNNTBPEModel.restore_from(restore_path=str(model_path), map_location=device)
    model.eval()
    model.to(device)
    # Use autocast for fp16 inference; RNNT supports mixed precision on T4
    amp_dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32
    return model, amp_dtype


# ======== Data loading =======================================================
@dataclass
class Item:
    audio_path: Path
    text: str
    duration: float


def load_manifest(manifest_path: Path, max_minutes: float | None, max_utts: int | None) -> List[Item]:
    items: List[Item] = []
    total_sec = 0.0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            ap = Path(j["audio_filepath"]) if "audio_filepath" in j else Path(j["audio"].get("path", ""))
            txt = j.get("text", j.get("transcript", ""))
            dur = float(j.get("duration", 0.0))
            items.append(Item(audio_path=ap, text=txt, duration=dur))
            total_sec += dur
            if max_utts and len(items) >= max_utts:
                break
            if max_minutes and total_sec / 60.0 >= max_minutes:
                break
    return items


# ======== Audio helpers ======================================================

def load_and_prepare(wav_path: Path, target_sr: int = SAMPLE_RATE):
    wav, sr = torchaudio.load(str(wav_path))  # shape [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    wav = wav.squeeze(0).numpy()
    return wav, sr


def write_wav_bytesio(samples: np.ndarray, sr: int) -> bytes:
    """Encode float32 PCM to WAV bytes in-memory (16-bit PCM)."""
    with io.BytesIO() as bio:
        sf.write(bio, samples, sr, subtype="PCM_16", format="WAV")
        return bio.getvalue()


# ======== Streaming loop =====================================================
@dataclass
class UtteranceMetrics:
    utt_id: str
    chunk_ms: int
    use_fp16: bool
    audio_sec: float
    wer: float
    rtf: float
    first_partial_ms: Optional[float]
    finalization_ms: Optional[float]
    gpu_peak_gb: float


def stream_decode_utt(
    model: EncDecRNNTBPEModel,
    audio: np.ndarray,
    ref_text: str,
    sr: int,
    chunk_ms: int,
    amp_dtype: torch.dtype,
    device: str,
) -> UtteranceMetrics:
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr}"
    samples_per_chunk = int(sr * (chunk_ms / 1000.0))
    num_chunks = max(1, math.ceil(len(audio) / samples_per_chunk))

    # Timing + GPU stats
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
    t0 = time.perf_counter()

    # Partial tracking
    partial_text = ""
    first_partial_ms = None
    accumulated_audio = np.zeros(0, dtype=np.float32)

    # For RTF, measure total compute time spent in model.transcribe
    total_decode_time = 0.0

    # We'll reuse a NamedTemporaryFile path to avoid repeated open/close cost
    tmpdir = tempfile.mkdtemp(prefix="asr_stream_")
    tmpwav = Path(tmpdir) / "accum.wav"

    try:
        for ci in range(num_chunks):
            start = ci * samples_per_chunk
            end = min(len(audio), (ci + 1) * samples_per_chunk)
            accumulated_audio = np.concatenate([accumulated_audio, audio[start:end]])

            if (ci + 1) % PARTIAL_EVERY_N_CHUNKS != 0 and ci != num_chunks - 1:
                # Skip decoding this step to reduce overhead
                if SIMULATE_REALTIME:
                    # sleep the actual chunk duration to mimic wall clock
                    time.sleep((end - start) / sr)
                continue

            # Write/overwrite the current buffer as a WAV file in-place
            wav_bytes = write_wav_bytesio(accumulated_audio, sr)
            with open(tmpwav, "wb") as f:
                f.write(wav_bytes)

            # Decode the current buffer
            step_t0 = time.perf_counter()
            with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"), dtype=amp_dtype, enabled=(amp_dtype==torch.float16)):
                hyp_list = model.transcribe(paths2audio_files=[str(tmpwav)], batch_size=1)
            if device == "cuda":
                torch.cuda.synchronize()
            step_dt = time.perf_counter() - step_t0
            total_decode_time += step_dt

            hyp = hyp_list[0] if isinstance(hyp_list, list) else str(hyp_list)

            # First partial detection
            if first_partial_ms is None and hyp.strip():
                first_partial_ms = (time.perf_counter() - t0) * 1000.0

            partial_text = hyp

            if SIMULATE_REALTIME:
                # sleep to emulate real-time
                time.sleep((end - start) / sr)

        # After final chunk, we consider finalization latency = final decode runtime
        finalization_ms = step_dt * 1000.0 if num_chunks > 0 else 0.0

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    t1 = time.perf_counter()
    wall_time = t1 - t0

    hyp_final = partial_text
    ref = ref_text or ""
    curr_wer = wer(ref, hyp_final) if ref else float("nan")

    # RTF uses *compute time* (decoding) over audio duration; also report wallclock via logs
    audio_sec = len(audio) / sr
    rtf = total_decode_time / max(audio_sec, 1e-6)

    if device == "cuda":
        gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        gpu_peak_gb = 0.0

    return UtteranceMetrics(
        utt_id=str(hash((len(audio), ref[:32]))),
        chunk_ms=chunk_ms,
        use_fp16=(amp_dtype == torch.float16),
        audio_sec=audio_sec,
        wer=curr_wer,
        rtf=rtf,
        first_partial_ms=first_partial_ms,
        finalization_ms=finalization_ms,
        gpu_peak_gb=gpu_peak_gb,
    )


# ======== Runner =============================================================

def run():
    if DEVICE == "cuda":
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        print("GPU:", torch.cuda.get_device_name(CUDA_DEVICE_INDEX))
    else:
        print("Warning: CUDA not available; results will not reflect T4 behavior.")

    model, amp_dtype = load_model(MODEL_PATH, device=DEVICE, use_fp16=USE_FP16)
    items = load_manifest(MANIFEST_PATH, max_minutes=MAX_TOTAL_AUDIO_MINUTES, max_utts=MAX_UTTS)
    print(f"Loaded {len(items)} utterances from manifest: {MANIFEST_PATH}")

    per_utt: List[UtteranceMetrics] = []
    for it in tqdm(items, desc="Streaming"):
        audio, sr = load_and_prepare(it.audio_path, SAMPLE_RATE)
        m = stream_decode_utt(model, audio, it.text, sr, CHUNK_MS, amp_dtype, DEVICE)
        per_utt.append(m)

    wers = [m.wer for m in per_utt if not math.isnan(m.wer)]
    rtf_vals = [m.rtf for m in per_utt]
    firsts = [m.first_partial_ms for m in per_utt if m.first_partial_ms is not None]
    finals = [m.finalization_ms for m in per_utt if m.finalization_ms is not None]
    peaks = [m.gpu_peak_gb for m in per_utt]

    summary = {
        "chunk_ms": CHUNK_MS,
        "use_fp16": USE_FP16,
        "utt_count": len(per_utt),
        "audio_min": sum(m.audio_sec for m in per_utt) / 60.0,
        "wer_mean": float(np.mean(wers)) if wers else float("nan"),
        "rtf_p50": float(np.percentile(rtf_vals, 50)) if rtf_vals else float("nan"),
        "rtf_p95": float(np.percentile(rtf_vals, 95)) if rtf_vals else float("nan"),
        "first_partial_ms_p50": float(np.percentile(firsts, 50)) if firsts else float("nan"),
        "first_partial_ms_p95": float(np.percentile(firsts, 95)) if firsts else float("nan"),
        "finalization_ms_p50": float(np.percentile(finals, 50)) if finals else float("nan"),
        "finalization_ms_p95": float(np.percentile(finals, 95)) if finals else float("nan"),
        "gpu_peak_gb_p95": float(np.percentile(peaks, 95)) if peaks else float("nan"),
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    per_rows = [asdict(m) for m in per_utt]
    for r in per_rows:
        r["use_fp16"] = int(r["use_fp16"])
    write_csv(PER_UTT_CSV, per_rows)
    write_csv(SUMMARY_CSV, [summary])

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
