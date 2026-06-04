"""
Embedding encoder. Two interchangeable backends, picked by config.yml
(`model.embedding.runtime: pytorch | onnx`).

Public API stays the same regardless of backend:

    encode(texts: list[str]) -> np.ndarray   # shape (N, dim), L2-normalised
    model_version() -> str                   # the EMBEDDING_MODEL_ID stamp

`model_version()` is what gets written into chunk_embeddings.model_version,
so a wrong runtime/quantization is detectable in the database.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config


def model_version() -> str:
    return config.EMBEDDING_MODEL_ID


# ── PyTorch backend (dev + initial production) ───────────────────────────────
_pt_model = None


def _get_pt_model():
    """Lazy-loaded sentence-transformers model. Used when runtime != 'onnx'."""
    global _pt_model
    if _pt_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading SciNCL via sentence-transformers '{config.SCINCL_MODEL}' …")
        _pt_model = SentenceTransformer(
            config.SCINCL_MODEL,
            cache_folder=config.SCINCL_CACHE_DIR,
        )
        print("  Model ready (pytorch).")
    return _pt_model


def _encode_pytorch(texts: list[str]) -> np.ndarray:
    return _get_pt_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


# ── ONNX backend (lightweight production) ─────────────────────────────────────
_ort_session = None
_ort_tokenizer = None


def _cls_pool(last_hidden: np.ndarray) -> np.ndarray:
    """SciNCL uses [CLS]-token pooling (verified via Pooling module config:
    pooling_mode_cls_token=True). [CLS] is always position 0."""
    return last_hidden[:, 0, :]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _verify_manifest(onnx_path: Path):
    """If a MANIFEST.json sits next to the ONNX file, sanity-check it.
    Required: manifest.id matches config.EMBEDDING_MODEL_ID and manifest.dim
    matches config.EMBEDDING_DIM. This is a cheap guard against shipping the
    wrong artifact."""
    manifest_path = onnx_path.parent / "MANIFEST.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("id") != config.EMBEDDING_MODEL_ID:
        raise RuntimeError(
            f"ONNX manifest id={manifest.get('id')!r} does not match "
            f"config.EMBEDDING_MODEL_ID={config.EMBEDDING_MODEL_ID!r}. "
            "Either update config.yml or re-export with the right id."
        )
    if int(manifest.get("dim", -1)) != config.EMBEDDING_DIM:
        raise RuntimeError(
            f"ONNX manifest dim={manifest.get('dim')!r} does not match "
            f"config.EMBEDDING_DIM={config.EMBEDDING_DIM}."
        )


def _get_ort():
    global _ort_session, _ort_tokenizer
    if _ort_session is not None:
        return _ort_session, _ort_tokenizer

    if not config.EMBEDDING_ONNX_PATH:
        raise RuntimeError(
            "model.embedding.runtime is 'onnx' but model.embedding.onnx_path is "
            "not set. Run scripts/export_scincl_onnx.py and update config.yml."
        )
    onnx_path = Path(config.EMBEDDING_ONNX_PATH)
    if not onnx_path.exists():
        raise RuntimeError(
            f"ONNX model not found at {onnx_path}. Run scripts/export_scincl_onnx.py "
            "and ensure the file ships with the production image."
        )

    _verify_manifest(onnx_path)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    print(f"  Loading SciNCL ONNX from {onnx_path} …")
    _ort_session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    _ort_tokenizer = AutoTokenizer.from_pretrained(str(onnx_path.parent))
    print("  Model ready (onnx).")
    return _ort_session, _ort_tokenizer


def _encode_onnx(texts: list[str]) -> np.ndarray:
    sess, tok = _get_ort()
    inputs = tok(texts, padding=True, truncation=True, return_tensors="np", max_length=512)
    feeds = {
        "input_ids":      inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs.get(
            "token_type_ids",
            np.zeros_like(inputs["input_ids"]),
        ).astype(np.int64),
    }
    last_hidden = sess.run(["last_hidden_state"], feeds)[0]
    return _l2_normalize(_cls_pool(last_hidden)).astype(np.float32)


# ── Public encode() dispatcher ────────────────────────────────────────────────
def encode(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings → (N, EMBEDDING_DIM) float32 array, L2-normalised
    so cosine similarity == dot product.

    Backend chosen by config.EMBEDDING_MODEL_RUNTIME.
    """
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
    if config.EMBEDDING_MODEL_RUNTIME == "onnx":
        return _encode_onnx(texts)
    return _encode_pytorch(texts)


# Kept for callers that previously imported get_model() directly.
def get_model():
    return _get_pt_model()
