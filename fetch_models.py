#!/usr/bin/env python3
"""
Fetch and pin ASR models, then optionally export to ONNX and TensorRT.

- Downloads pretrained checkpoints from Hugging Face (pinned revision)
- Saves REVISION + CHECKSUMS.sha256 for reproducibility
- (Optional) Exports NeMo .nemo -> ONNX
- (Optional) Builds TensorRT FP16 engine from ONNX via `trtexec`

Edit the CONFIG section below to choose models and export options.
"""

# ==========================
# CONFIG — EDIT THIS SECTION
# ==========================
MODELS = {
    # name: (repo_id, revision)
    "en": ("nvidia/stt_en_conformer_transducer_large", "main"),
    # Optional comparisons:
    # "en_streaming": ("nvidia/stt_en_fastconformer_hybrid_large_streaming_multi", "main"),
    # "de": ("nvidia/stt_de_conformer_transducer_large", "main"),
}

OUTPUT_DIR = "models"
FORCE = False     # re-download even if REVISION matches

# Export options (apply to all models found with a .nemo)
EXPORT_ONNX = False
ONNX_OPSET = 17   # 17 is a solid default for recent ONNX/ORT/TRT

EXPORT_TRT = False      # set True to build TensorRT engine
TRT_FP16 = True         # FP16 recommended on T4
TRT_WORKSPACE_MB = 4096 # increase if builder runs OOM
TRT_EXTRA_ARGS = []     # e.g. shapes/optimization profiles if needed
# ==========================

import hashlib
import os
import sys
import subprocess
from pathlib import Path

# Compatible import across hub versions
try:
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.utils import HfHubError  # newer hubs
    except Exception:
        HfHubError = Exception
except Exception:
    print(
        "Missing or incompatible huggingface_hub. Install/update with:\n"
        "  pip install -U 'huggingface_hub>=0.21'\n"
        "Optional (faster downloads):\n"
        "  pip install -U 'hf-transfer'  # then set HF_HUB_ENABLE_HF_TRANSFER=1",
        file=sys.stderr,
    )
    raise


def sha256_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path) -> None:
    """Write SHA256 checksums for all files under root to CHECKSUMS.sha256."""
    checksum_path = root / "CHECKSUMS.sha256"
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() or p == checksum_path:
            continue
        rel = p.relative_to(root)
        digest = sha256_file(p)
        lines.append(f"{digest}  {rel.as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def already_downloaded(target_dir: Path, expected_revision: str) -> bool:
    rev_file = target_dir / "REVISION"
    if not rev_file.exists():
        return False
    actual = rev_file.read_text(encoding="utf-8").strip()
    return actual == expected_revision


def fetch_one(name: str, repo_id: str, revision: str, outdir: Path, force: bool) -> Path:
    target = outdir / name
    target.mkdir(parents=True, exist_ok=True)

    if already_downloaded(target, revision) and not force:
        print(f"[skip] {name}: already at revision {revision}")
        return target

    print(f"[download] {name}: {repo_id}@{revision}")
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target.as_posix(),
        )
    except HfHubError as e:
        print(f"[error] Failed to download {repo_id}@{revision}: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[error] Unexpected failure downloading {repo_id}@{revision}: {e}", file=sys.stderr)
        sys.exit(3)

    (target / "REVISION").write_text(revision + "\n", encoding="utf-8")
    write_checksums(target)

    print(f"[ok] {name} → {target}")
    return target


# ---------- Export helpers ----------

def ensure_nemo_available():
    try:
        import nemo.collections.asr as nemo_asr  # noqa: F401
    except Exception:
        print(
            "NeMo is required to export .nemo to ONNX.\n"
            "Install a CUDA-matched stack, e.g.:\n"
            "  pip install 'nemo_toolkit[asr]>=1.23.0' onnx onnxruntime-gpu\n"
            "Make sure your PyTorch/torchvision/torchaudio match your CUDA.",
            file=sys.stderr,
        )
        raise


def export_nemo_to_onnx(nemo_path: Path, out_dir: Path, opset: int) -> Path:
    import nemo.collections.asr as nemo_asr
    print(f"[load] {nemo_path.name}")
    model = nemo_asr.models.ASRModel.restore_from(nemo_path.as_posix(), map_location="cpu")
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    print(f"[export] ONNX -> {onnx_path} (opset={opset})")
    model.export(onnx_path.as_posix(), opset_version=opset)
    try:
        import onnx
        m = onnx.load(onnx_path.as_posix())
        io = [v.name for v in list(m.graph.input) + list(m.graph.output)]
        print("[info] ONNX IO:", io)
    except Exception as e:
        print("[warn] Could not inspect ONNX:", e)
    return onnx_path


def build_trt_engine(onnx_path: Path, plan_path: Path, fp16: bool, workspace_mb: int, extra_args=None):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path.as_posix()}",
        f"--saveEngine={plan_path.as_posix()}",
        f"--workspace={workspace_mb}",
        "--buildOnly",
    ]
    if fp16:
        cmd.append("--fp16")
    if extra_args:
        cmd.extend(extra_args)
    print("[build] TensorRT:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ERROR: trtexec not found. Install TensorRT or run inside a TensorRT container.", file=sys.stderr)
        sys.exit(4)
    print(f"[ok] TensorRT plan -> {plan_path}")


def maybe_export(model_dir: Path):
    # find a .nemo
    nemo_files = list(model_dir.glob("*.nemo"))
    if not nemo_files:
        print(f"[info] No .nemo found in {model_dir}; skipping export.")
        return

    nemo_path = nemo_files[0]
    export_dir = model_dir / "export"

    onnx_path = None
    if EXPORT_ONNX:
        ensure_nemo_available()
        onnx_path = export_nemo_to_onnx(nemo_path, export_dir, ONNX_OPSET)

    if EXPORT_TRT:
        if onnx_path is None:
            # user may have an existing ONNX already
            onnx_path = export_dir / "model.onnx"
            if not onnx_path.exists():
                print("ERROR: ONNX not found for TRT build. Enable EXPORT_ONNX or place model.onnx in export/", file=sys.stderr)
                sys.exit(5)
        plan_path = export_dir / ("model_fp16.plan" if TRT_FP16 else "model.plan")
        build_trt_engine(onnx_path, plan_path, TRT_FP16, TRT_WORKSPACE_MB, TRT_EXTRA_ARGS)

    # Update checksums to include exported artifacts
    write_checksums(model_dir)


def main():
    outdir = Path(OUTPUT_DIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads if hf-transfer installed

    for name, (repo_id, revision) in MODELS.items():
        model_dir = fetch_one(name, repo_id, revision, outdir, FORCE)
        maybe_export(model_dir)

    print("\nAll done. Models are ready.\n")


if __name__ == "__main__":
    main()