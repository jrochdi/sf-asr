#!/usr/bin/env python3
"""
Streaming ASR micro-prototype (simulator).

- Loads a NeMo Conformer-Transducer (.nemo) and runs greedy decoding
- Simulates streaming: 20 ms frames; emits partials every CHUNK_MS
- Endpointing: energy-VAD + hypothesis stability
- Reports: WER (jiwer), RTF p50/p95, first-partial latency, finalize latency, peak GPU memory

This is a simple, single-file prototype to satisfy the assignment.
You can later swap the ENGINE to ONNX/TensorRT for lower latency.
"""

# ==========================
# CONFIG — EDIT THIS SECTION
# ==========================
ENGINE = "nemo"   # "nemo" (default). Later: "onnx" when you export.
NEMO_PATH = "models/en/stt_en_conformer_transducer_large.nemo"

# Data manifests (Common Voice eval provided by the assignment)
MANIFESTS = [
    "data/asr_eval/common_voice_17_0/manifest_en.jsonl",  # edit if your paths differ
    # "data/asr_eval/common_voice/de/manifest.jsonl",
]

SAMPLE_RATE = 16000
FRAME_MS = 20                   # 20 ms frames
CHUNK_MS = 100                  # run decode every 100 ms
ROLLING_WINDOW_S = 1.2          # seconds of context sent to decoder each tick
LOOKAHEAD_MS = 160              # “algorithmic” look-ahead you allow (for budgeting)

# Endpointing
VAD_FRAME_MS = 20
VAD_SIL_THRESH_DB = -45.0       # energy threshold (dBFS) for silence
END_MIN_SIL_MS = 300            # require >=300 ms silence
STABILITY_MS = 250              # no hypothesis change for 250 ms → finalize

# Decode / precision
USE_FP16 = True                 # autocast fp16 for NeMo
BATCH_SIZE = 1                  # single stream only (challenge requirement)

# Reporting
MAX_AUDIO_MINUTES = 15.0        # process up to 15 min total audio
SEED = 42
# ==========================

import json, time, math, gc, statistics, os, sys
from pathlib import Path
import numpy as np
import tempfile
import soundfile as sf

# Compatibility shim for newer huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "ModelFilter"):
        class _Dummy:
            def __init__(self, *a, **k): pass
        huggingface_hub.ModelFilter = _Dummy
except ImportError:
    pass

# Torch / NeMo (lazy import so script can still run without GPU if needed)
import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# jiwer for WER
try:
    from jiwer import wer
except ImportError:
    print("Install jiwer: pip install jiwer", file=sys.stderr)
    sys.exit(1)

# Audio I/O
try:
    import soundfile as sf
except ImportError:
    print("Install soundfile: pip install soundfile", file=sys.stderr)
    sys.exit(1)

# Optional resample
try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False


# -------------------------
# Utility: data loading
# -------------------------
def read_manifest(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Common Voice style: obj["audio_filepath"], obj["text"]
            wav = obj.get("audio_filepath") or obj.get("audio", None)
            txt = obj.get("text") or obj.get("sentence") or ""
            if wav is None:
                continue
            yield wav, txt


def load_audio_16k(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        if not HAVE_TORCHAUDIO:
            raise RuntimeError("Install torchaudio to resample to 16 kHz")
        wav = torch.from_numpy(audio).unsqueeze(0)
        res = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        audio = res.squeeze(0).numpy()
    return audio


# -------------------------
# Simple energy VAD
# -------------------------
def frame_energy_dbfs(frame):
    rms = np.sqrt(np.mean(np.square(frame)) + 1e-12)
    db = 20.0 * math.log10(rms + 1e-9)
    return db

def vad_silence_mask(wav, sr, frame_ms, thresh_db):
    n = int(sr * frame_ms / 1000)
    n = max(n, 1)
    m = len(wav) // n
    mask = np.zeros(m, dtype=bool)
    for i in range(m):
        fr = wav[i*n:(i+1)*n]
        mask[i] = (frame_energy_dbfs(fr) < thresh_db)
    return mask  # True = silent


# -------------------------
# Model wrappers
# -------------------------
class NemoRNNT:
    def __init__(self, nemo_path: str, use_fp16: bool = True):
        import nemo.collections.asr as nemo_asr
        print(f"[load] NeMo RNNT: {nemo_path}")
        self.model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=device)
        self.model.eval().to(device)
        self.use_fp16 = use_fp16 and device.type == "cuda"
        # greedy decoding config (default; small beam optional)
        if hasattr(self.model, "change_decoding_strategy"):
            self.model.change_decoding_strategy(decoding_cfg={"greedy": {"max_symbols_per_step": 10}})
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    @torch.inference_mode()
    def decode_chunk(self, audio_16k: np.ndarray):
        """
        Greedy decode a short audio snippet and return text.
        Uses a temp WAV + NeMo's file-based transcribe() to stay compatible
        across NeMo versions that don't accept raw tensors.
        """
        # ensure float32 mono at 16 kHz
        audio = audio_16k.astype(np.float32)

        # write to a temp WAV and call transcribe(paths2audio_files=[...])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, 16000, subtype="PCM_16")
            # NeMo transcribe expects a list of paths
            hyps = self.model.transcribe(
                paths2audio_files=[tmp.name],
                return_hypotheses=True,
                batch_size=1,
            )

        # hyps is a list with one Hypothesis (or list-of-list depending on NeMo)
        hyp = hyps[0]
        if isinstance(hyp, list):  # some versions return [[Hypothesis]]
            hyp = hyp[0]
        return getattr(hyp, "text", "")


# -------------------------
# Streaming simulator
# -------------------------
def simulate_stream(model, wav: np.ndarray, ref_text: str, sr: int):
    """
    Feed 20 ms frames; every CHUNK_MS we decode the last ROLLING_WINDOW_S.
    Measure:
      - first partial latency
      - finalize latency after endpoint
      - total compute time (for RTF)
    """
    frame_len = int(sr * FRAME_MS / 1000)
    chunk_frames = CHUNK_MS // FRAME_MS
    window_len = int(sr * ROLLING_WINDOW_S)

    # VAD for endpoints
    sil_mask = vad_silence_mask(wav, sr, VAD_FRAME_MS, VAD_SIL_THRESH_DB)
    sil_run = 0
    endpoint_ms = None

    # Streaming loop timestamps
    t0_audio = 0.0                       # audio "arrival" time at start (ms)
    first_partial_ms = None
    last_emit_ms = 0
    last_partial = ""
    last_change_ms = None
    finalized_text = None

    # Metrics
    compute_ms = 0.0

    # Process frames
    num_frames = len(wav) // frame_len
    for fidx in range(num_frames):
        # audio t advances by FRAME_MS per frame
        audio_time_ms = (fidx + 1) * FRAME_MS

        # update VAD silence streak
        if fidx < len(sil_mask) and sil_mask[fidx]:
            sil_run += VAD_FRAME_MS
        else:
            sil_run = 0

        # decode cadence: every CHUNK_MS
        if (fidx + 1) % chunk_frames == 0 and finalized_text is None:
            beg = max(0, int(fidx * frame_len + frame_len) - window_len)
            cur_chunk = wav[beg : fidx * frame_len + frame_len]

            t1 = time.perf_counter()
            partial = model.decode_chunk(cur_chunk)
            t2 = time.perf_counter()
            dt_ms = (t2 - t1) * 1000.0
            compute_ms += dt_ms
            last_emit_ms = audio_time_ms

            # first partial timestamp
            if partial and not first_partial_ms:
                first_partial_ms = audio_time_ms  # wallclock since start of utterance

            # track stability
            if partial != last_partial:
                last_partial = partial
                last_change_ms = audio_time_ms

            # endpoint rule: >= END_MIN_SIL_MS silence + hypothesis stable for STABILITY_MS
            if sil_run >= END_MIN_SIL_MS and last_change_ms is not None:
                endpoint_ms = audio_time_ms - sil_run  # when silence started
                if (audio_time_ms - last_change_ms) >= STABILITY_MS:
                    finalized_text = last_partial

    # Finalize if not done (tail utterance)
    if finalized_text is None:
        finalized_text = last_partial
        endpoint_ms = endpoint_ms or (num_frames * FRAME_MS)

    finalize_latency_ms = max(0.0, (last_emit_ms - endpoint_ms)) if endpoint_ms is not None else 0.0

    # WER for this utterance
    sample_wer = wer(ref_text.lower(), (finalized_text or "").lower())

    return {
        "first_partial_ms": float(first_partial_ms or 0.0),
        "finalize_ms": float(finalize_latency_ms),
        "compute_ms": float(compute_ms),
        "audio_ms": float(num_frames * FRAME_MS),
        "wer": float(sample_wer),
        "final_text": finalized_text or "",
    }


def main():
    # limit total minutes if needed
    np.random.seed(SEED)

    # load model
    if ENGINE == "nemo":
        model = NemoRNNT(NEMO_PATH, use_fp16=USE_FP16)
    else:
        print("Only ENGINE='nemo' implemented in this prototype. Export to ONNX to add ENGINE='onnx'.")
        sys.exit(1)

    # iterate dataset
    utterance_metrics = []
    total_audio_s = 0.0

    for mani in MANIFESTS:
        if not Path(mani).exists():
            print(f"[warn] manifest not found: {mani}")
            continue
        for wav_path, ref in read_manifest(mani):
            wav = load_audio_16k(wav_path)
            dur_s = len(wav) / SAMPLE_RATE
            if dur_s <= 0.05:
                continue
            m = simulate_stream(model, wav, ref, SAMPLE_RATE)
            utterance_metrics.append(m)
            total_audio_s += dur_s
            if total_audio_s / 60.0 >= MAX_AUDIO_MINUTES:
                break
        if total_audio_s / 60.0 >= MAX_AUDIO_MINUTES:
            break

    # aggregate metrics
    if not utterance_metrics:
        print("No audio processed.")
        return

    rtf_list = [(m["compute_ms"] / 1000.0) / (m["audio_ms"] / 1000.0) for m in utterance_metrics]
    fp_list = [m["first_partial_ms"] for m in utterance_metrics if m["first_partial_ms"] > 0]
    fin_list = [m["finalize_ms"] for m in utterance_metrics]

    rtf_p50 = statistics.median(rtf_list)
    rtf_p95 = np.percentile(rtf_list, 95)
    fp_p50 = statistics.median(fp_list) if fp_list else 0.0
    fp_p95 = np.percentile(fp_list, 95) if fp_list else 0.0
    fin_p50 = statistics.median(fin_list)
    fin_p95 = np.percentile(fin_list, 95)

    overall_wer = statistics.mean([m["wer"] for m in utterance_metrics])

    # GPU mem
    peak_mem_gb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    # print report
    print("\n=== Streaming ASR Report ===")
    print(f"Audio processed: {total_audio_s/60.0:.2f} min")
    print(f"WER: {overall_wer:.3f}")
    print(f"RTF p50 / p95: {rtf_p50:.3f} / {rtf_p95:.3f}")
    print(f"First-partial (ms) p50 / p95: {fp_p50:.0f} / {fp_p95:.0f}")
    print(f"Finalize after endpoint (ms) p50 / p95: {fin_p50:.0f} / {fin_p95:.0f}")
    print(f"Peak GPU memory (GB): {peak_mem_gb:.2f}")

    # simple CSV dump
    out = Path("results_stream_eval.csv")
    with out.open("w", encoding="utf-8") as f:
        f.write("file,wer,first_partial_ms,finalize_ms,rtf,final_text\n")
        # (we didn’t track file names in this minimal version; you can extend simulate_stream to include them)
        for m in utterance_metrics:
            rtf = (m["compute_ms"]/1000.0) / (m["audio_ms"]/1000.0)
            f.write(f"NA,{m['wer']:.4f},{m['first_partial_ms']:.1f},{m['finalize_ms']:.1f},{rtf:.4f},\"{m['final_text'].replace('\"','') }\"\n")
    print(f"Saved per-utterance metrics to {out}")

    # cleanup
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()