# Low-Latency Streaming ASR – Submission

This repository contains my submission for the take-home exercise described in **TASK.md**.  
It includes a micro-prototype implementation, design document, and evaluation results.
The Design document and results/discussion are in `document.pdf`.

## Setup

Create the conda environment:

```
conda env create -f environment.yml
conda activate takehome-asr310
```

## Run Simulation

To reproduce the streaming simulation on the Common Voice dataset:

```
python simulate.py
```

This will generate per-utterance and summary CSV files with latency and accuracy metrics.

