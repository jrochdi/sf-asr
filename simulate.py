import copy
import time
import pyaudio as pa
import numpy as np
import torch, torchaudio
import json
from tqdm import tqdm
import csv
import pynvml
import webrtcvad

from omegaconf import OmegaConf, open_dict
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# sample rate of audio
SAMPLE_RATE = 16000 # Hz

# Which pretrained model to use
model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"

# Set lookahead size to use, can be 0, 80, 480 or 1040 ms
lookahead_size = 0

# Which decoder to use, can be "rnnt" or "ctc"
decoder_type = "rnnt"

# Encoder step length is 80 ms for FastConformer models by design
ENCODER_STEP_LENGTH = 80 

# Simulated (virtual) pacing: don't sleep, just advance a virtual clock
USE_VIRTUAL_PACING = False

# For onset detection in stream simulation
USE_ONLINE_VAD  = True     
VAD_AGGRESSIVENESS = 3        # 0..3 (3 = most strict)
SUBFRAME_MS = 20              # WebRTC VAD supports 10/20/30 ms
VAD_MIN_SUBFRAMES = 4         # need N consecutive speech subframes to tentatively trigger
VAD_CONFIRM_MS = 400          # if no token within this window, revoke tentative onset

# Load model and apply configuration
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
if lookahead_size not in [0, 80, 480, 1040]:
    raise ValueError(
        f"specified lookahead_size {lookahead_size} is not one of the "
        "allowed lookaheads (can select 0, 80, 480 or 1040 ms)"
    )
left_context_size = asr_model.encoder.att_context_size[0]
asr_model.encoder.set_default_att_context_size([left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)])
asr_model.change_decoding_strategy(decoder_type=decoder_type)
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    # save time by doing greedy decoding and not trying to record the alignments
    decoding_cfg.strategy = "greedy"
    decoding_cfg.preserve_alignments = False
    if hasattr(asr_model, 'joint'):  # if an RNNT model
        # restrict max_symbols to make sure not stuck in infinite loop
        decoding_cfg.greedy.max_symbols = 10
        # sensible default parameter, but not necessary since batch size is 1
        decoding_cfg.fused_batch_size = -1
    asr_model.change_decoding_strategy(decoding_cfg)
asr_model.eval()

# Manifest 
MANIFEST = "data/asr_eval/common_voice_17_0/manifest_en.jsonl"
with open(MANIFEST, "r") as f:
    entries = [json.loads(line) for line in f]

# Normalizer for fair WER
norm = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

# get parameters to use as the initial cache state
cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
    batch_size=1
)

# init params we will use for streaming
previous_hypotheses = None
pred_out_stream = None
step_num = 0
pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
# cache-aware models require some small section of the previous processed_signal to
# be fed in at each timestep - we initialize this to a tensor filled with zeros
# so that we will do zero-padding for the very first chunk(s)
num_channels = asr_model.cfg.preprocessor.features
cache_pre_encode = torch.zeros((1, num_channels, pre_encode_cache_size), device=asr_model.device)

# helper function for extracting transcriptions
def extract_transcriptions(hyps):
    """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions

# define functions to init audio preprocessor and to
# preprocess the audio (ie obtain the mel-spectrogram)
def init_preprocessor(asr_model):
    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"
    
    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(asr_model.device)
    
    return preprocessor

preprocessor = init_preprocessor(asr_model)

def preprocess_audio(audio, asr_model):
    device = asr_model.device

    # doing audio preprocessing
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    processed_signal, processed_signal_length = preprocessor(
        input_signal=audio_signal, length=audio_signal_len
    )
    return processed_signal, processed_signal_length

def transcribe_chunk(new_chunk):
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num
    global cache_pre_encode
    
    # new_chunk int16 -> float32 in [-1,1]
    audio_data = new_chunk.astype(np.float32) / 32768.0

    # preprocessing timing (mel)
    t0 = time.time()  # wall clock; used only for per-chunk preprocessing time
    processed_signal, processed_signal_length = preprocess_audio(audio_data, asr_model)
    pre_ms = (time.time() - t0) * 1000.0

    processed_signal = processed_signal.to(dtype=torch.float32) 
    cache_pre_encode = cache_pre_encode.to(dtype=torch.float32) 

    # prepend cache
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[1]
    cache_pre_encode = processed_signal[:, :, -pre_encode_cache_size:]

    # model step timing (encoder+decoder)
    t1 = time.time()  # wall clock; used only for per-chunk model step time
    with torch.inference_mode():
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = asr_model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )
    step_ms = (time.time() - t1) * 1000.0

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    step_num += 1
    return final_streaming_tran[0], pre_ms, step_ms

### STREAMING BELOW ###

# calculate chunk_size
chunk_size = lookahead_size + ENCODER_STEP_LENGTH
frames_per_buffer = int(SAMPLE_RATE * chunk_size / 1000) - 1  
chunk_dur_s = frames_per_buffer / float(SAMPLE_RATE) 

# helper to get the virtual time in for virtual pacing
def virtual_now(metrics_state):
    # virtual clock for onset, first token, falls back to wall clock if disabled
    if USE_VIRTUAL_PACING and metrics_state.get("virt_now") is not None:
        return metrics_state["virt_now"]
    return time.time()

def callback(in_data, frame_count, time_info, status):
    signal = np.frombuffer(in_data, dtype=np.int16)
    if len(signal) < frame_count:
        signal = np.pad(signal, (0, frame_count - len(signal)), mode='constant')
    text = transcribe_chunk_metrics_wrapper(signal) # not printing during eval to avoid terminal spam
    return (in_data, pa.paContinue)

def feed_wav_into_callback(path, callback_fn):
    """
    Streams an arbitrary WAV (any sample rate / channel count) into the callback.
    Converts on the fly to 16 kHz, mono, int16 so the rest of the pipeline stays unchanged. 
    """
    waveform, sr = torchaudio.load(path)  # float32 tensor
    if waveform.ndim != 2:
        raise ValueError(f"Expected 2D tensor [channels, samples], got shape {tuple(waveform.shape)}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, N]

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

    samples = waveform.squeeze(0).clamp_(-1.0, 1.0)  # [N]

    int16_samples = (samples * 32768.0).to(torch.int16).cpu().numpy()  # dtype=int16

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    sub_len = int(SAMPLE_RATE * SUBFRAME_MS / 1000)  # samples per 20 ms

    total = int16_samples.shape[0]
    idx = 0

    # Pacing
    if USE_VIRTUAL_PACING:
        metrics_state["virt_now"] = 0.0 # virtual clock current time for stream (seconds)
    first_chunk = True

    while idx < total:
        # advance virtual time to "arrival" of this chunk 
        if USE_VIRTUAL_PACING:
            metrics_state["virt_now"] += chunk_dur_s

        chunk = int16_samples[idx : idx + frames_per_buffer]

        # If no vad, use the start of first buffer as onset so first-partial works
        if (not USE_ONLINE_VAD) and first_chunk and metrics_state["t_speech_onset"] is None:
            # onset is start-of-buffer time: one chunk earlier than its "arrival"
            metrics_state["t_speech_onset"] = metrics_state["virt_now"] - chunk_dur_s if USE_VIRTUAL_PACING else (time.time() - chunk_dur_s)
        first_chunk = False

        # Detect last chunk for finalization latency
        is_last = (idx + frames_per_buffer) >= total
        if is_last and metrics_state.get("t_endpoint_seen") is None:
            metrics_state["t_endpoint_seen"] = virtual_now(metrics_state)   # virtual time when last audio chunk is observed
            metrics_state["t_endpoint_wall"] = time.time()                  # wall time when last audio chunk is observed (used for finalization)

        if chunk.shape[0] < frames_per_buffer:
            chunk = np.pad(chunk, (0, frames_per_buffer - chunk.shape[0]), mode='constant')

        # ---------- online vad ----------
        if USE_ONLINE_VAD and metrics_state["t_first_partial"] is None:
            consec = metrics_state["vad_consec_subframes"]
            n_full = (len(chunk) // sub_len) * sub_len
            if n_full > 0:
                view = chunk[:n_full].reshape(-1, sub_len)
                virt_chunk_start = metrics_state["virt_now"] - chunk_dur_s if USE_VIRTUAL_PACING else None

                for k, sf in enumerate(view):
                    is_speech = vad.is_speech(sf.tobytes(), SAMPLE_RATE)
                    if is_speech:
                        consec += 1
                        if metrics_state["t_speech_onset_tentative"] is None and consec >= VAD_MIN_SUBFRAMES:
                            # tentative onset time
                            if virt_chunk_start is not None:
                                subframe_offset_s = (k + 1) * (SUBFRAME_MS / 1000.0)
                                onset_time = virt_chunk_start + subframe_offset_s
                            else:
                                onset_time = virtual_now(metrics_state)
                            metrics_state["t_speech_onset_tentative"] = onset_time  # tentative onset awaiting confirmation
                    else:
                        consec = 0

                    # Debounce: if no token within confirm window, revoke tentative onset
                    t0 = metrics_state["t_speech_onset_tentative"]
                    if (t0 is not None) and (metrics_state["t_first_partial"] is None):
                        now_m = virtual_now(metrics_state)
                        if (now_m - t0) * 1000.0 > VAD_CONFIRM_MS:
                            metrics_state["t_speech_onset_tentative"] = None
                            consec = 0

                metrics_state["vad_consec_subframes"] = consec
        # ------------------------------------------

        data_bytes = chunk.tobytes()
        _ = callback_fn(data_bytes, frames_per_buffer, None, None)
        idx += frames_per_buffer

def reset_stream_state():
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num
    global cache_pre_encode

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=1
    )
    previous_hypotheses = None
    pred_out_stream = None
    step_num = 0

    num_channels = int(asr_model.cfg.preprocessor.features)
    cache_pre_encode = torch.zeros((1, num_channels, pre_encode_cache_size), device=asr_model.device)

# GPU mem tracker
pynvml.nvmlInit()
_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Aggregates
all_rtf, all_first_partial_ms, all_finalization = [], [], []
all_wers_raw, all_wers_norm, all_durations = [], [], []
no_partial_count = 0
results_rows = []
peak_mem_used = 0
all_med_pre_ms, all_med_step_ms = [], []

# per-file mutable state, set before each file
metrics_state = {
    "t_start_overall": None,             # stream start time on the clock used for RTF (virtual if pacing, else wall)
    "t_speech_onset": None,              # confirmed speech onset time (virtual) used as zero for first-partial
    "t_speech_onset_tentative": None,    # tentative onset time from VAD (virtual) awaiting confirmation
    "t_first_partial": None,             # time of first emitted token (virtual)
    "last_text": "",
    "chunk_pre_ms": [],
    "chunk_step_ms": [],
    "t_endpoint_seen": None,             # time when last audio chunk observed (virtual) kept for reference
    "t_endpoint_wall": None,             # wall time when last audio chunk observed used for finalization
    "vad_consec_subframes": 0,
    "virt_now": None,                    # virtual clock current time for stream (seconds) advances by chunk_dur_s
}

def transcribe_chunk_metrics_wrapper(sig):
    out = transcribe_chunk(sig)
    if isinstance(out, tuple):
        out_text, pre_ms, step_ms = out
    else:
        out_text, pre_ms, step_ms = out, float("nan"), float("nan")

    if out_text and len(out_text.strip()) > 0 and metrics_state["t_first_partial"] is None:
        metrics_state["t_first_partial"] = virtual_now(metrics_state)  # first partial token time (virtual)

        # Confirm vad tentative onset, if any. If none, and vad disabled, use first_partial as onset fallback
        if metrics_state["t_speech_onset"] is None:
            if metrics_state["t_speech_onset_tentative"] is not None:
                metrics_state["t_speech_onset"] = metrics_state["t_speech_onset_tentative"]
            elif not USE_ONLINE_VAD:
                metrics_state["t_speech_onset"] = metrics_state["t_first_partial"]

    metrics_state["last_text"] = out_text
    metrics_state["chunk_pre_ms"].append(pre_ms)
    metrics_state["chunk_step_ms"].append(step_ms)

    meminfo = pynvml.nvmlDeviceGetMemoryInfo(_handle)
    global peak_mem_used
    peak_mem_used = max(peak_mem_used, meminfo.used)

    return out_text

# Loop over utterances of dataset
for entry in tqdm(entries):
    wav_path = entry["audio_filepath"]
    ref = entry.get("text", "")

    reset_stream_state()

    info = torchaudio.info(wav_path)
    audio_dur = float(info.num_frames) / float(info.sample_rate)
    all_durations.append(audio_dur)

    # start of stream 
    if USE_VIRTUAL_PACING:
        metrics_state["virt_now"] = 0.0 # virtual clock current time (seconds)
        metrics_state["t_start_overall"] = 0.0 # virtual start time (seconds)
    else:
        metrics_state["virt_now"] = None
        metrics_state["t_start_overall"] = time.time() # wall start time (seconds)

    metrics_state["t_first_partial"] = None
    metrics_state["t_speech_onset"] = None 
    metrics_state["t_speech_onset_tentative"] = None  
    metrics_state["last_text"] = ""
    metrics_state["chunk_pre_ms"] = []   
    metrics_state["chunk_step_ms"] = []         
    metrics_state["t_endpoint_seen"] = None
    metrics_state["t_endpoint_wall"] = None
    metrics_state["vad_consec_subframes"] = 0

    # run streaming
    feed_wav_into_callback(wav_path, callback)

    # end of stream times
    t_end_virtual = virtual_now(metrics_state) if USE_VIRTUAL_PACING else time.time()  # end time on RTF clock
    t_end_wall = time.time()                                                          # wall end time (used for finalization)

    # metrics
    if metrics_state["t_first_partial"] is None or metrics_state["t_speech_onset"] is None:
        first_partial_ms = np.nan
        no_partial_count += 1
    else:
        first_partial_ms = (metrics_state["t_first_partial"] - metrics_state["t_speech_onset"]) * 1000.0

    # finalization uses wall clock 
    if metrics_state.get("t_endpoint_wall") is not None:
        finalization_ms = (t_end_wall - metrics_state["t_endpoint_wall"]) * 1000.0
    else:
        finalization_ms = np.nan

    # RTF uses the same clock used to start (virtual if pacing, else wall)
    elapsed = (t_end_virtual - metrics_state["t_start_overall"])
    rtf = elapsed / max(audio_dur, 1e-9)

    hyp = metrics_state["last_text"]
    hyp_norm = norm(hyp) if hyp else ""
    if metrics_state["chunk_pre_ms"]:
        med_pre = float(np.median(metrics_state["chunk_pre_ms"]))
        med_step = float(np.median(metrics_state["chunk_step_ms"]))
    else:
        med_pre = float("nan")
        med_step = float("nan")

    all_first_partial_ms.append(first_partial_ms)
    all_finalization.append(finalization_ms)
    all_rtf.append(rtf)
    all_wers_raw.append(wer(ref, hyp))
    all_wers_norm.append(wer(norm(ref), hyp_norm))
    all_med_pre_ms.append(med_pre)
    all_med_step_ms.append(med_step)

    results_rows.append({
        "audio_filepath": wav_path,
        "ref": ref,
        "hyp_norm": hyp_norm,
        "hyp": hyp,
        "wer_norm": (wer(norm(ref), hyp_norm) if ref else np.nan),
        "duration_sec": audio_dur,
        "rtf": rtf,
        "first_partial_ms": first_partial_ms,
        "finalization_ms": finalization_ms,
        "median_pre_ms": med_pre,
        "median_model_step_ms": med_step,
    })

# Summary and csv
fp = np.array(all_first_partial_ms, dtype=float)
fp_valid = fp[~np.isnan(fp)]

print("\n=== Overall Metrics ===")
total_audio_min = sum(all_durations)/60.0
rtf_median = float(np.median(all_rtf))
rtf_p95 = float(np.percentile(all_rtf, 95))
if fp_valid.size:
    first_partial_median = float(np.median(fp_valid))
    first_partial_p95 = float(np.percentile(fp_valid, 95))
else:
    first_partial_median = None
    first_partial_p95 = None
print(f"Total audio: {total_audio_min:.1f} min")
print(f"RTF median / p95: {rtf_median:.3f} / {rtf_p95:.3f}")
if fp_valid.size:
    print(f"First partial median / p95 (ms): {first_partial_median:.1f} / {first_partial_p95:.1f}")
else:
    print("First partial median / p95 (ms): N/A (no partials detected)")
print(f"First-partial missing: {no_partial_count} / {len(all_first_partial_ms)}")
finalization_median = float(np.nanmedian(all_finalization))
print(f"Finalization median (ms): {finalization_median:.1f}")
if all_wers_raw:
    wer_raw_avg = float(np.mean(all_wers_raw))
    wer_norm_avg = float(np.mean(all_wers_norm))
    print(f"WER raw avg: {wer_raw_avg:.3f}")
    print(f"WER norm avg: {wer_norm_avg:.3f}")
else:
    wer_raw_avg = None
    wer_norm_avg = None
peak_gpu_gb = float(peak_mem_used/(1024**3))
print(f"Peak GPU Mem: {peak_gpu_gb:.2f} GB")
if all_med_pre_ms and all_med_step_ms:
    pre_med = float(np.nanmedian(all_med_pre_ms))
    pre_p95 = float(np.nanpercentile(all_med_pre_ms,95))
    step_med = float(np.nanmedian(all_med_step_ms))
    step_p95 = float(np.nanpercentile(all_med_step_ms,95))
    print(f"Preprocessing median / p95 (ms): {pre_med:.1f} / {pre_p95:.1f}")
    print(f"Model step median / p95 (ms): {step_med:.1f} / {step_p95:.1f}")
else:
    pre_med = pre_p95 = step_med = step_p95 = None

# Save per-utterance CSV
csv_path = f"streaming_results_{decoder_type}_{lookahead_size}ms.csv"
fieldnames = [
    "audio_filepath","ref","hyp_norm","hyp","wer_norm",
    "duration_sec","rtf","first_partial_ms","finalization_ms",
    "median_pre_ms","median_model_step_ms"
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results_rows:
        writer.writerow(row)
print(f"\nSaved per-utterance results to: {csv_path}")

# Save summary CSV
summary_csv_path = f"streaming_summary_{decoder_type}_{lookahead_size}ms.csv"
summary_fieldnames = [
    "model_name","decoder_type","lookahead_ms","encoder_step_ms","precision",
    "num_utterances","total_audio_min",
    "rtf_p50","rtf_p95",
    "first_partial_p50","first_partial_p95","first_partial_missing",
    "finalization_p50",
    "wer_raw_avg","wer_norm_avg",
    "preprocess_p50","preprocess_p95",
    "model_step_p50","model_step_p95",
    "peak_gpu_mem_gb"
]

summary_row = {
    "model_name": model_name,
    "decoder_type": decoder_type,
    "lookahead_ms": lookahead_size,
    "encoder_step_ms": ENCODER_STEP_LENGTH,
    "num_utterances": len(results_rows),
    "total_audio_min": total_audio_min,
    "rtf_p50": rtf_median,
    "rtf_p95": rtf_p95,
    "first_partial_p50": first_partial_median,
    "first_partial_p95": first_partial_p95,
    "first_partial_missing": no_partial_count,
    "finalization_p50": finalization_median,
    "wer_raw_avg": wer_raw_avg,
    "wer_norm_avg": wer_norm_avg,
    "preprocess_p50": pre_med,
    "preprocess_p95": pre_p95,
    "model_step_p50": step_med,
    "model_step_p95": step_p95,
    "peak_gpu_mem_gb": peak_gpu_gb
}

with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
    writer.writeheader()
    writer.writerow(summary_row)

print(f"Saved summary (CSV) to: {summary_csv_path}")