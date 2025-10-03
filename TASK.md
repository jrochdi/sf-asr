# Streaming ASR on a T4 GPU — Design & Justify Your Architecture

This repository is a take‑home exercise focused on designing a low‑latency streaming ASR system that runs on a single NVIDIA T4 GPU. It includes a small Common Voice evaluation set and JSONL manifests to help you prototype, measure latency/accuracy trade‑offs, and justify your architectural choices.

No model training is needed.

At a glance:
- **Data**: Common Voice clips and manifests under `data/asr_eval/common_voice`
- **Languages**: **English** (required) and **German** (nice‑to‑have)
- **Constraints**: Server‑side GPU only (T4), mixed‑precision allowed (FP16/INT8)
- **Deliverables**: Design document, micro‑prototype, one ablation, results & discussion

---

## Dataset
- **Path**: `data/asr_eval/common_voice`
- **Languages**:
  - English
  - German

## Constraints
- **Target hardware**: 1× NVIDIA T4 (cloud)
- **Precision**: Mixed‑precision allowed (FP16/INT8)
- **Scope**: Server‑side GPU only; no on‑device and no CPU baselines
- **Use case**: Real‑time voice chat (English required; German nice‑to‑have)

## Latency and Throughput Targets (single active stream)
- **First partial token**: ≤ 300 ms from audio arrival
- **Finalization after endpoint**: ≤ 150 ms
- **Throughput**: RTF ≤ 0.2×

## What to Hand In
- **Design Document** (2–3 pages)
  - Model family and latency budget by stage (feature stack → acoustic model → decoder/LM → endpointing)
  - Interruption handling for barge‑in
- **Micro‑prototype Python code and experiment results** (< 8 hr of coding)
  - A single Python script (no server needed) that simulates real-time streaming (feed 20–40 ms frames) for 10–15 min of public speech
  - For minimum latency in Triton: set request concurrency = 1, disable dynamic batching, prefer a single model instance
  - Report: WER via jiwer, RTF p50/p95, first‑partial latency, GPU memory usage
- **One ablation experiment** (exact ablation depends on your design choice)
- **One‑page Results & Discussion**
  - Table: Model / Precision / Engine → WER, RTF p50/p95, First‑partial (ms), Peak GPU‑Mem (GB)
  - 4–6 bullets: what drove latency; what surprised you; what you’d try next
