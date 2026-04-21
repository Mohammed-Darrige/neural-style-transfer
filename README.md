# Neural Style Transfer

Public release of a LoRA-based neural style transfer project built on Stable Diffusion 1.5.

## Overview

**Architecture Status:** The Text-to-Image pipeline is fully active and operational. The Image-to-Image module is currently locked and undergoing a major architecture refactoring.

This repository contains:

- a Kaggle-ready training notebook
- four trained LoRA adapters
- a FastAPI inference backend
- a sample showcase image

Supported styles:

- Cubism
- Pop Art
- Post-Impressionism
- Ukiyo-e

## Training Summary

- Base model: `runwayml/stable-diffusion-v1-5`
- Training environment: Kaggle
- Images per style: 500
- LoRA rank: 8
- Max train steps: 600
- Resolution: 512
- Optimizer mode: 8-bit Adam
- Adapter size: about 6.12 MB each

## Repository Structure

- `Artoria_Style_Transfer_Kaggle_Training.ipynb`: end-to-end Kaggle training notebook
- `weights/`: trained LoRA adapters for all four styles
- `backend/`: FastAPI inference service
- `requirements.txt`: notebook-side training dependencies
- `run_summary.json`: cleaned training summary
- `artoria_style_transfer_showcase.png`: sample output grid

## Local Backend Setup

Run from the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The backend loads adapters from `weights/lora-output-<style>/pytorch_lora_weights.safetensors`.

## Included Weights

- `weights/lora-output-cubism/`
- `weights/lora-output-pop-art/`
- `weights/lora-output-post-impressionism/`
- `weights/lora-output-ukiyo-e/`

## Notes

- Training logs are excluded from git.
- The notebook is the main training artifact.
