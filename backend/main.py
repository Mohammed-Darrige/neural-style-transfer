import base64
import binascii
import os
import random
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal

import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel

app = FastAPI(title="Neural Style Transfer API")
pipeline: StableDiffusionPipeline | None = None
i2i_pipeline: StableDiffusionImg2ImgPipeline | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_LOCK = Lock()

SUPPORTED_STYLES = ("cubism", "post-impressionism", "ukiyo-e")
WEIGHTS_ROOT = Path(
    os.getenv(
        "STYLE_WEIGHTS_ROOT",
        str(Path(__file__).resolve().parent.parent / "weights"),
    )
)
INFERENCE_STEPS = int(os.getenv("STYLE_INFERENCE_STEPS", "30"))
GUIDANCE_SCALE = float(os.getenv("STYLE_GUIDANCE_SCALE", "7.5"))
IMAGE_SIZE = int(os.getenv("STYLE_IMAGE_SIZE", "512"))
T2I_GUIDANCE_SCALE = float(os.getenv("STYLE_T2I_GUIDANCE_SCALE", str(GUIDANCE_SCALE)))
I2I_INFERENCE_STEPS = int(os.getenv("STYLE_I2I_INFERENCE_STEPS", "30"))
I2I_MIN_DIM = int(os.getenv("STYLE_I2I_MIN_DIM", str(IMAGE_SIZE)))
I2I_MAX_DIM = int(os.getenv("STYLE_I2I_MAX_DIM", "768"))

DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "STYLE_NEGATIVE_PROMPT",
    "photograph, photorealistic, realism, real life, low quality, worst quality, blurry, jpeg artifacts, watermark, text, signature",
)

STYLE_PRESETS: dict[str, dict[str, object]] = {
    "cubism": {
        "training_caption": "in cubism style, geometric shapes, picasso style",
        "negative_prompt": "3d render, cg",
        "i2i_strength": 0.65,
        "i2i_guidance_scale": 8.0,
    },
    "post-impressionism": {
        "training_caption": "in post-impressionism style, painting style",
        "negative_prompt": "3d render, cg",
        "i2i_strength": 0.65,
        "i2i_guidance_scale": 8.0,
    },
    "ukiyo-e": {
        "training_caption": "in ukiyo-e style, woodblock print, flat colors",
        "negative_prompt": "3d render, cg",
        "i2i_strength": 0.65,
        "i2i_guidance_scale": 8.0,
    },
}

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StyleRequest(BaseModel):
    style_type: Literal["cubism", "post-impressionism", "ukiyo-e"]
    prompt: str | None = None
    init_image: str | None = None
    strength: float | None = None
    seed: int | None = None


def _resolve_lora_path(style: str) -> Path:
    candidate = (
        WEIGHTS_ROOT / f"lora-output-{style}" / "pytorch_lora_weights.safetensors"
    )
    if not candidate.exists():
        raise HTTPException(
            status_code=404,
            detail=f"LoRA weights not found for '{style}' at '{candidate}'.",
        )
    return candidate


def _decode_init_image(payload: str) -> Image.Image:
    try:
        init_payload = payload.split(",", 1)[1] if "," in payload else payload
        init_bytes = base64.b64decode(init_payload, validate=True)
        init_img = Image.open(BytesIO(init_bytes))
        return ImageOps.exif_transpose(init_img).convert("RGB")
    except (binascii.Error, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid source image.") from exc


def _round_to_multiple(value: float, multiple: int = 8) -> int:
    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


def _prepare_init_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Source image has invalid dimensions.")

    short_side = min(width, height)
    long_side = max(width, height)
    scale = I2I_MIN_DIM / short_side

    if long_side * scale > I2I_MAX_DIM:
        scale = I2I_MAX_DIM / long_side

    resized_width = _round_to_multiple(width * scale)
    resized_height = _round_to_multiple(height * scale)

    return image.resize((resized_width, resized_height), RESAMPLE)


def _compose_prompt(
    style: str,
    prompt: str | None,
    is_img2img: bool,
) -> tuple[str, str]:
    preset = STYLE_PRESETS[style]
    negative_prompt = f"{DEFAULT_NEGATIVE_PROMPT}, {preset['negative_prompt']}"
    base_prompt = (prompt or "").strip()

    if is_img2img:
        caption = str(preset["training_caption"])
        parts: list[str] = [caption]
        if base_prompt:
            parts.append(base_prompt)
        return ", ".join(parts), negative_prompt

    t2i_caption = str(preset.get("t2i_prompt", preset["training_caption"]))
    if not base_prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt is required for text-to-image mode.",
        )

    return f"{base_prompt}, {t2i_caption}", negative_prompt


def _resolve_strength(requested: float | None, default: float) -> float:
    if requested is None:
        return default
    if requested < 0.1 or requested > 1.0:
        raise HTTPException(
            status_code=422,
            detail="Strength must be between 0.1 and 1.0.",
        )
    return requested


@app.on_event("startup")
async def load_pipeline() -> None:
    global pipeline, i2i_pipeline

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("Loading base stable diffusion pipeline from Hugging Face...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading fp16-safe VAE (sd-vae-ft-mse)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch_dtype,
    ).to(device)

    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        vae=vae,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    i2i_pipeline = StableDiffusionImg2ImgPipeline(**pipeline.components)
    i2i_pipeline = i2i_pipeline.to(device)
    i2i_pipeline.set_progress_bar_config(disable=True)

    assert id(pipeline.unet) == id(i2i_pipeline.unet), "UNet not shared between pipelines"

    print(
        f"Pipeline loaded. device={device}"
    )


@app.post("/generate")
async def generate_style(request: StyleRequest):
    global pipeline, i2i_pipeline

    if pipeline is None or i2i_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    folder_style = request.style_type.lower()
    lora_path = _resolve_lora_path(folder_style)
    preset = STYLE_PRESETS[folder_style]
    is_img2img = request.init_image is not None

    seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(
        "[StyleRequest]",
        {"style_type": request.style_type, "prompt": request.prompt, "has_init_image": is_img2img, "seed": seed},
    )

    image_buffer = BytesIO()
    init_img: Image.Image | None = None

    try:
        with INFERENCE_LOCK:
            if is_img2img:
                init_source = _decode_init_image(request.init_image or "")
                init_img = _prepare_init_image(init_source)

            enhanced_prompt, negative_prompt = _compose_prompt(
                folder_style,
                request.prompt,
                is_img2img,
            )

            print(f"[ResolvedPrompt] {enhanced_prompt}")
            print(
                f"[Params] seed={seed}"
                + (
                    f" i2i_strength={request.strength if request.strength is not None else preset['i2i_strength']}"
                    f" guidance={preset['i2i_guidance_scale']}"
                    f" steps={I2I_INFERENCE_STEPS}"
                    if is_img2img
                    else f" guidance={T2I_GUIDANCE_SCALE}"
                )
            )

            pipeline.load_lora_weights(str(lora_path))

            with torch.inference_mode():
                if is_img2img:
                    strength = _resolve_strength(
                        request.strength,
                        float(preset["i2i_strength"]),
                    )
                    result = i2i_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=init_img,
                        strength=strength,
                        guidance_scale=float(preset["i2i_guidance_scale"]),
                        num_inference_steps=I2I_INFERENCE_STEPS,
                        generator=generator,
                    )
                else:
                    result = pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=T2I_GUIDANCE_SCALE,
                        num_inference_steps=INFERENCE_STEPS,
                        width=IMAGE_SIZE,
                        height=IMAGE_SIZE,
                        generator=generator,
                    )

            image = result[0][0] if isinstance(result, tuple) else result.images[0]

            image.save(image_buffer, format="JPEG")
            image_bytes = image_buffer.getvalue()

            return Response(content=image_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        ) from exc
    finally:
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
