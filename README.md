# Neural Style Transfer

LoRA-based neural style transfer project built on Stable Diffusion 1.5.

## Overview

This project fine-tunes separate LoRA adapters for four art styles:

- Cubism
- Pop Art
- Post-Impressionism
- Ukiyo-e

The training pipeline was prepared for Kaggle and optimized for constrained GPU environments. The project also includes a FastAPI inference backend for text-to-image and image-to-image generation.

## Training Summary

- Base model: `runwayml/stable-diffusion-v1-5`
- Training environment: Kaggle
- Images per style: 500
- LoRA rank: 8
- Max train steps: 600
- Resolution: 512
- Optimizer mode: 8-bit Adam
- Output size per adapter: about 6.12 MB

## Repository Structure

- `Artoria_Style_Transfer_Kaggle_Training.ipynb`: end-to-end Kaggle training notebook
- `backend/`: FastAPI inference service
- `requirements.txt`: notebook-side training dependencies
- `run_summary.json`: exported training summary
- `artoria_style_transfer_showcase.png`: example output grid

## Local Backend Setup

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The backend expects LoRA weights under `backend/weights/`, but those weights are intentionally not committed.

## Notes

- Model weights are excluded from git.
- Training logs are excluded from git.
- The Kaggle notebook is the primary training artifact.
