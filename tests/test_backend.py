"""Lightweight backend & artifact verification tests (no GPU model download required)."""
from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import (
    I2I_MAX_DIM,
    I2I_MIN_DIM,
    MAX_IMAGE_BYTES,
    _decode_init_image,
    _prepare_init_image,
    _resolve_strength,
    app,
)

ROOT = Path(__file__).resolve().parents[1]


def test_weights_manifest_sha256_integrity():
    manifest_path = ROOT / "weights" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapters = manifest if isinstance(manifest, list) else manifest.get("adapters", manifest)
    assert len(adapters) >= 3

    for style in ("cubism", "post-impressionism", "ukiyo-e"):
        weights_file = ROOT / "weights" / f"lora-output-{style}" / "pytorch_lora_weights.safetensors"
        assert weights_file.exists(), f"Missing adapter: {weights_file}"
        actual_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()
        # Find matching entry in manifest
        matched = [
            entry for key, entry in (adapters.items() if isinstance(adapters, dict) else enumerate(adapters))
            if isinstance(entry, dict) and entry.get("sha256") == actual_sha
        ]
        assert matched, f"SHA-256 mismatch for {style}: {actual_sha}"


def test_decode_init_image_limits():
    with pytest.raises(HTTPException) as exc_bad:
        _decode_init_image("not-valid-base64!!!")
    assert exc_bad.value.status_code == 400

    oversized = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode("ascii")
    with pytest.raises(HTTPException) as exc_big:
        _decode_init_image(oversized)
    assert exc_big.value.status_code == 413

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="PNG")
    valid_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    img = _decode_init_image(valid_b64)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_prepare_init_image_and_strength_bounds():
    img = Image.new("RGB", (1024, 512))
    prepared = _prepare_init_image(img)
    w, h = prepared.size
    assert w % 8 == 0 and h % 8 == 0
    assert (w, h) == (768, 384)
    assert max(w, h) <= I2I_MAX_DIM

    sq = _prepare_init_image(Image.new("RGB", (256, 256)))
    assert sq.size == (I2I_MIN_DIM, I2I_MIN_DIM)

    with pytest.raises(HTTPException) as exc_low:
        _resolve_strength("cubism", -0.1, 0.65)
    assert exc_low.value.status_code == 422

    s_min = _resolve_strength("ukiyo-e", 0.0, 0.65)
    s_max = _resolve_strength("ukiyo-e", 1.0, 0.65)
    assert 0.42 <= s_min <= s_max <= 0.62


def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False
        assert data["supported_styles"] == ["cubism", "post-impressionism", "ukiyo-e"]
        assert all(data["weights_present"].values())
