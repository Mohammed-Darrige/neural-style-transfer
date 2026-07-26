# Neural Style Transfer

Stable Diffusion 1.5 image-to-image style transfer with verified LoRA artifacts, a FastAPI backend, and a reproducible Colab training workflow.

## Deployed Adapters

The repository ships three rank-4 UNet LoRA adapters:

| Style | File size | SHA-256 |
|---|---:|---|
| Cubism | 3,226,184 bytes (3.08 MiB) | `2a68d54f9d4e4d2341bfb4637500aa0750dc623b9c82da60da8ebf49dcdcb45e` |
| Post-Impressionism | 3,226,184 bytes (3.08 MiB) | `ae70f996a25278bb6e9a9eac1d09bbd909552ab99266f91f5ba38213ad933acf` |
| Ukiyo-e | 3,226,184 bytes (3.08 MiB) | `35237cd93e20156729f59088f61e61e088543f6e51d097b838a18e2b6c33bcc1` |

The files contain 256 FP32 LoRA tensors, target SD 1.x UNet attention projections (`to_k`, `to_q`, `to_v`, `to_out.0`), and have inferred rank 4. The complete manifest is `weights/manifest.json`.

Pop Art was trained in the same four-style run but is intentionally archived and not exposed by the current backend or live UI.

## Verified Training Provenance

The shipped rank-4 adapters were created by the multi-style Google Colab workflow documented in:

- `Artoria_Colab_MultiStyle_LoRA_Training.ipynb`
- `TRAINING_PROVENANCE.md`

Verified run configuration:

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
| Data preparation | center crop and random horizontal flip |

The original run prepared 1,167 Ukiyo-e, 2,235 Cubism, 1,483 Pop Art, and 6,310 Post-Impressionism images. These are prepared corpus counts, not a claim that every image was consumed in a fixed number of epochs.

## Inference

The backend supports text-to-image and image-to-image requests. The portfolio primarily uses image-to-image mode.

- Base model: `runwayml/stable-diffusion-v1-5`
- Img2img scheduler: Euler Ancestral
- Default local text-to-image steps: 30
- Default img2img steps: 50
- Img2img strength input: `0.0` to `1.0`

The user-facing style control maps nonlinearly to both img2img denoising strength and LoRA scale. It is not a direct Diffusers denoise-strength value. See the deployed backend implementation for the exact mapping.

## Local Backend Setup

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The backend loads each adapter from:

```text
weights/lora-output-<style>/pytorch_lora_weights.safetensors
```

Visual output can vary across hardware and library revisions. Record the backend revision and request parameters when comparing results.

## Repository Layout

- `Artoria_Colab_MultiStyle_LoRA_Training.ipynb`: clean executable version of the verified multi-style training workflow
- `TRAINING_PROVENANCE.md`: artifact, ZIP, hash, and worker-script evidence
- `archive/Artoria_Style_Transfer_Kaggle_Training_rank8_legacy.ipynb`: retained historical rank-8 reconstruction; it did not produce the shipped rank-4 binaries
- `weights/`: shipped adapter binaries and manifest
- `backend/`: FastAPI and Modal deployment code
- `run_summary.json`: machine-readable training and artifact summary

## Scope and Limitations

- The adapter files prove rank, tensor structure, dtype, and hashes. They do not independently encode dataset identity, optimizer, training steps, or image count.
- The Colab workflow downloaded Diffusers from the moving `main` branch. The provenance document records the closest reconstructable worker-script commit and hash.
- Runtime hashes from the live Modal container require a deployment metadata endpoint before they can be asserted as cryptographically verified.
