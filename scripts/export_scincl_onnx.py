"""
Export SciNCL (malteos/scincl) → ONNX INT8 + write a signed manifest.

Run once on a dev machine (PyTorch + sentence-transformers must be installed).
The resulting artifact is what production loads — no PyTorch needed at serve.

Output layout (committed to models/ or uploaded to a release):
    models/
      scincl-int8.onnx          ← quantized model (~110 MB)
      tokenizer.json            ← HuggingFace fast tokenizer
      tokenizer_config.json
      special_tokens_map.json
      vocab.txt
      MANIFEST.json             ← {id, dim, sha256, baseline_cosines, exported_at}

How parity is verified
----------------------
Before writing the manifest, this script runs a golden-set check:

    for text in GOLDEN_TEXTS:
        pytorch_vec = sentence_transformers.encode(text)   # FP32 baseline
        onnx_vec    = ort_session.run(text)                # INT8 candidate
        cosine      = pytorch_vec · onnx_vec

It writes the manifest ONLY if every cosine ≥ MIN_COSINE (default 0.998).
This is what guarantees the swap is safe: same model, same pooling, same
normalization — the only delta is INT8 quantization, and we measure it.

Usage:
    python scripts/export_scincl_onnx.py
    python scripts/export_scincl_onnx.py --outdir models/ --min-cosine 0.995
    python scripts/export_scincl_onnx.py --skip-quantize     # FP32 ONNX
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config


GOLDEN_TEXTS = [
    "Mechanical Engineering [SEP] Laplace Transforms: Integral transform used to "
    "convert differential equations into algebraic equations.",
    "Electrical Engineering [SEP] RLC Circuit: Series or parallel arrangement of a "
    "resistor, inductor, and capacitor whose dynamics obey a second-order ODE.",
    "Fluid Mechanics [SEP] Navier-Stokes Equations: Partial differential equations "
    "describing the motion of viscous fluid substances.",
    "Computer Science [SEP] Dynamic Programming: Algorithmic technique that solves "
    "problems by combining solutions to overlapping subproblems.",
    "Materials Science [SEP] Phase Diagrams: Charts showing the thermodynamically "
    "stable phases of a material as functions of temperature and composition.",
    "Thermodynamics [SEP] Carnot Cycle: Idealised thermodynamic cycle that "
    "represents the maximum efficiency of a heat engine.",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    denom  = np.maximum(mask.sum(axis=1), 1e-9)
    return summed / denom


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",        default="models")
    ap.add_argument("--hf-name",       default=config.SCINCL_MODEL)
    ap.add_argument("--min-cosine",    type=float, default=0.998)
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load the PyTorch model (only on the dev machine running this) ──
    print(f"[1/5] Loading {args.hf_name} via sentence-transformers (FP32 baseline) …")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch

    st_model  = SentenceTransformer(args.hf_name, cache_folder=config.SCINCL_CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_name, cache_dir=config.SCINCL_CACHE_DIR)
    hf_model  = AutoModel.from_pretrained(args.hf_name, cache_dir=config.SCINCL_CACHE_DIR)
    hf_model.eval()

    # ── Step 2: ONNX export of the bare transformer ────────────────────────────
    print(f"[2/5] Exporting transformer to ONNX …")
    onnx_fp32 = outdir / "scincl-fp32.onnx"
    dummy = tokenizer("hello world", return_tensors="pt")
    torch.onnx.export(
        hf_model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(onnx_fp32),
        input_names  = ["input_ids", "attention_mask", "token_type_ids"],
        output_names = ["last_hidden_state"],
        dynamic_axes = {
            "input_ids":         {0: "batch", 1: "seq"},
            "attention_mask":    {0: "batch", 1: "seq"},
            "token_type_ids":    {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version = 17,
        do_constant_folding = True,
    )

    # ── Step 3: optional INT8 dynamic quantization ─────────────────────────────
    if args.skip_quantize:
        onnx_final = onnx_fp32
        model_id   = "scincl-v1-fp32"
    else:
        print(f"[3/5] INT8 dynamic-quantizing …")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        onnx_final = outdir / "scincl-int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_fp32),
            model_output=str(onnx_final),
            weight_type=QuantType.QInt8,
        )
        onnx_fp32.unlink()
        model_id = "scincl-v1-int8"

    print(f"        wrote {onnx_final} ({onnx_final.stat().st_size / 1e6:.1f} MB)")

    # Save tokenizer next to the ONNX file so the encoder is fully self-contained.
    tokenizer.save_pretrained(outdir)

    # ── Step 4: parity check ───────────────────────────────────────────────────
    print(f"[4/5] Parity check against PyTorch baseline …")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_final), providers=["CPUExecutionProvider"])

    baseline = st_model.encode(GOLDEN_TEXTS, normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)

    inputs = tokenizer(GOLDEN_TEXTS, padding=True, truncation=True,
                       return_tensors="np", max_length=512)
    feeds = {
        "input_ids":      inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs.get(
            "token_type_ids",
            np.zeros_like(inputs["input_ids"]),
        ).astype(np.int64),
    }
    last_hidden = sess.run(["last_hidden_state"], feeds)[0]
    candidate = _l2(_mean_pool(last_hidden, inputs["attention_mask"]))

    cosines = [float(np.dot(baseline[i], candidate[i])) for i in range(len(GOLDEN_TEXTS))]
    worst   = min(cosines)
    print(f"        cosines: {[round(c, 4) for c in cosines]}")
    print(f"        worst = {worst:.4f}  (gate: {args.min_cosine})")

    if worst < args.min_cosine:
        print(f"✗ Parity FAILED. Refusing to write manifest.", file=sys.stderr)
        sys.exit(2)

    # ── Step 5: signed manifest ────────────────────────────────────────────────
    print(f"[5/5] Writing manifest …")
    manifest = {
        "id":             model_id,
        "hf_name":        args.hf_name,
        "dim":            int(baseline.shape[1]),
        "pooling":        "mean",
        "normalize":      "l2",
        "max_seq_len":    512,
        "sha256":         _sha256(onnx_final),
        "size_bytes":     onnx_final.stat().st_size,
        "exported_at":    int(time.time()),
        "parity": {
            "golden_size":      len(GOLDEN_TEXTS),
            "worst_cosine":     worst,
            "all_cosines":      cosines,
            "min_cosine_gate":  args.min_cosine,
        },
    }
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print("\n✓ ONNX export complete.")
    print(f"  Set in config.yml under model.embedding:")
    print(f"    runtime:   onnx")
    print(f"    onnx_path: {onnx_final}")
    print(f"    id:        {model_id}")


if __name__ == "__main__":
    main()
