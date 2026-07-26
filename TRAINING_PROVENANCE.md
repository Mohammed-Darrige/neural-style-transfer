# Training Provenance

## Scope

This document records the evidence chain for the three rank-4 adapters shipped in `weights/`. It distinguishes facts verified from binaries and archives from settings known only through the executed Colab notebook.

## Verified Artifact Chain

The preserved archive `final_lora_weights_only.zip` contains four adapters created in one sequential run:

| ZIP entry | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `Ukiyo-e/pytorch_lora_weights.safetensors` | 3,226,184 | `35237cd93e20156729f59088f61e61e088543f6e51d097b838a18e2b6c33bcc1` | deployed |
| `Cubism/pytorch_lora_weights.safetensors` | 3,226,184 | `2a68d54f9d4e4d2341bfb4637500aa0750dc623b9c82da60da8ebf49dcdcb45e` | deployed |
| `Pop_Art/pytorch_lora_weights.safetensors` | 3,226,184 | `26db6ac4bd363ad5bec9d0fbbac06b3ad0cf6bb7ff813a9d57f334dc8eabf219` | archived, not deployed |
| `Post-Impressionism/pytorch_lora_weights.safetensors` | 3,226,184 | `ae70f996a25278bb6e9a9eac1d09bbd909552ab99266f91f5ba38213ad933acf` | deployed |

The archive SHA-256 is:

```text
b0a081e56c7a3f4b7b2a689200119917d7dcded6b9adc515f0aba0047fbc5f1f
```

The archive ZIP timestamps are 2026-04-24 13:21:46, 13:49:18, 14:16:38, and 14:44:08 in the same order as the training loop. ZIP timestamps have no timezone field.

## Producer Notebook

The executed local source notebook was:

```text
Colab_MultiStyle_LoRA_Training11.ipynb
SHA-256: 3d640337ef5884bd008bdc8e366ec506d435f4f29a70d1dcd028a0f9a0e331b6
```

It prepared four style folders, trained them sequentially, then created both `all_lora_weights.zip` and `final_lora_weights_only.zip`. The latter has the exact archive hash above. `Artoria_Colab_MultiStyle_LoRA_Training.ipynb` is a clean, output-free public reconstruction of that workflow.

## Training Configuration Recorded by the Executed Notebook

| Setting | Value |
|---|---|
| Base model | `runwayml/stable-diffusion-v1-5` |
| Dataset | `h7alasaleh/artoria-dataset` downloaded from Kaggle |
| Dataset folders | `CUkiyo_e`, `CCubism`, `CPop_Art`, `CPost_Impressionism` |
| Resolution | 512 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Maximum steps | 1,000 per style |
| Learning rate | `1e-4` |
| Scheduler | cosine |
| Checkpoints | every 500 steps |
| Seed | 42 |
| Precision | fp16 |
| Augmentation | center crop and random horizontal flip |
| Prepared image counts | 1,167 Ukiyo-e; 2,235 Cubism; 1,483 Pop Art; 6,310 Post-Impressionism |

## Worker Script Boundary

The original notebook installed Diffusers from GitHub `main` and downloaded `examples/text_to_image/train_text_to_image_lora.py` from `main` without pinning a commit. The exact worker-script snapshot was not preserved.

The closest reconstructable repository state immediately before the recorded 2026-04-24 archive is:

```text
Diffusers commit: d0c9cbad28d7d3bba28db94622e13500c4179075
Worker SHA-256: 0377b3d93e0fc7e30f3fd8d5d18920a9d6bca81d7e719b1ed8d61973364339a4
```

That worker defaults to rank 4, sets `lora_alpha=args.rank`, targets `to_k`, `to_q`, `to_v`, and `to_out.0`, and saves Diffusers-format `pytorch_lora_weights.safetensors` files. These facts match the shipped tensor structure.

## What the Binaries Prove

- rank 4;
- 256 FP32 tensors and 797,184 parameters per adapter;
- UNet attention LoRA targets and no text-encoder LoRA tensors;
- file sizes and SHA-256 values.

The binaries alone do not encode the dataset, image count, optimizer, seed, training steps, scheduler, or exact base-model revision. Those values are provenance claims from the matching executed notebook, not safetensors metadata.
