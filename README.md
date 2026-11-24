# Neural Style Transfer

Stable Diffusion 1.5 text-to-image and image-to-image style transfer with verified rank-4 UNet LoRA adapters, a FastAPI inference backend, and a reproducible Colab training workflow.

- **Live Demo:** [darrige.tech/projects/neural-style-lab](https://darrige.tech/projects/neural-style-lab)

## Visual Showcase (Cubism LoRA Pilot — Seed 101)

Side-by-side comparison from our Kaggle Dual-T4 Cubism LoRA pilot (`seed=101`) comparing the unadapted base output against the reloaded LoRA adapter:

| Base Output (Unadapted) | Cubism LoRA Adapter Reloaded |
| :---: | :---: |
| ![Base output seed 101](assets/samples/cubism-base-seed101.png) | ![Cubism LoRA output seed 101](assets/samples/cubism-lora-seed101.png) |

## Deployed SD 1.5 Adapters

The repository ships three rank-4 UNet LoRA adapters (`weights/manifest.json`):

| Style | File size | SHA-256 |
|---|---:|---|
| Cubism | 3,226,184 bytes (3.08 MiB) | `2a68d54f9d4e4d2341bfb4637500aa0750dc623b9c82da60da8ebf49dcdcb45e` |
| Post-Impressionism | 3,226,184 bytes (3.08 MiB) | `ae70f996a25278bb6e9a9eac1d09bbd909552ab99266f91f5ba38213ad933acf` |
| Ukiyo-e | 3,226,184 bytes (3.08 MiB) | `35237cd93e20156729f59088f61e61e088543f6e51d097b838a18e2b6c33bcc1` |

Each file contains 256 FP32 LoRA tensors targeting SD 1.x UNet attention projections (`to_k`, `to_q`, `to_v`, `to_out.0`) with inferred rank 4. Pop Art was trained in the same four-style run and is archived without deployment.

## Verified Training Provenance

The shipped rank-4 adapters were produced by the multi-style Google Colab workflow documented in [`Artoria_Colab_MultiStyle_LoRA_Training.ipynb`](Artoria_Colab_MultiStyle_LoRA_Training.ipynb) and [`TRAINING_PROVENANCE.md`](TRAINING_PROVENANCE.md):

| Setting | Value |
|---|---|
| Base model | `runwayml/stable-diffusion-v1-5` |
| Styles trained | Ukiyo-e, Cubism, Pop Art, Post-Impressionism |
| Resolution | 512x512 |
| Steps | 1,000 per style |
| Batch size / accumulation | 1 / 4 |
| Learning rate | `1e-4` with cosine scheduling |
| Rank / alpha | 4 / 4 |
| Seed | 42 |
| Precision | fp16 |
| Data preparation | Center crop and random horizontal flip |

Prepared corpus counts: 1,167 Ukiyo-e, 2,235 Cubism, 1,483 Pop Art, and 6,310 Post-Impressionism images.

## Local Backend & API

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

- `GET /health` — Reports pipeline readiness, active device, supported styles, and adapter file presence.
- `POST /generate` — Accepts `style_type` (`cubism`, `post-impressionism`, `ukiyo-e`), optional `prompt`, `init_image` (base64), `strength` (`0.0`–`1.0`), and `seed`.
- **Input Limits:** Decoded `init_image` payloads are capped at **10 MB** (HTTP 413) and **4096x4096** pixels (~16.7 MP, HTTP 400).
- **Img2img Mapping:** User `strength` (`0.0`–`1.0`) maps nonlinearly (`u^1.6`) to denoising strength `[0.42, 0.62]` and LoRA scale `[0.95, 1.30]` with `EulerAncestralDiscreteScheduler` (50 steps).

## Verification

Run the lightweight backend & SHA-256 manifest tests without downloading GPU weights:

```bash
SKIP_MODEL_LOAD=1 pytest -v
```

## Repository Layout

- `backend/main.py`: FastAPI inference server with `lifespan` startup, upload limits, and `/health`.
- `weights/`: Shipped rank-4 `.safetensors` adapters and `manifest.json`.
- `assets/samples/`: Real pilot comparison images.
- `tests/test_backend.py`: Weight SHA-256 integrity, input validation, and API tests.
- `Artoria_Colab_MultiStyle_LoRA_Training.ipynb`: Executable multi-style Colab training notebook.
- `TRAINING_PROVENANCE.md` & `run_summary.json`: Provenance chain and metadata.

## Scope & Limitations

- Adapter `.safetensors` files verify rank, tensor layout, dtype, and SHA-256 hashes; they do not cryptographically encode dataset identity or optimizer state.
- Full GPU inference requires a CUDA-enabled environment or Modal/Colab runtime.
