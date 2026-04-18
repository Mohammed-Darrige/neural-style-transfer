import base64
import os
import random
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal

import torch
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
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

SUPPORTED_STYLES = ("cubism", "pop-art", "post-impressionism", "ukiyo-e")
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
    "low quality, worst quality, blurry, jpeg artifacts, watermark, text, signature",
)

STYLE_PRESETS: dict[str, dict[str, object]] = {
    "cubism": {
        "training_caption": "an artwork in the cubism style",
        "negative_prompt": "photograph, smooth skin, soft focus, jpeg artifacts",
        "lora_scale": 1.0,
        "i2i_strength": 0.55,
        "i2i_guidance_scale": 8.5,
    },
    "pop-art": {
        "training_caption": "an artwork in the pop art style",
        "negative_prompt": "photograph, sepia, grayscale, jpeg artifacts",
        "lora_scale": 1.0,
        "i2i_strength": 0.50,
        "i2i_guidance_scale": 8.5,
    },
    "post-impressionism": {
        "training_caption": "an artwork in the post-impressionism style",
        "negative_prompt": "photograph, flat digital shading, smooth blending, jpeg artifacts",
        "lora_scale": 1.0,
        "i2i_strength": 0.55,
        "i2i_guidance_scale": 8.5,
    },
    "ukiyo-e": {
        "training_caption": "an artwork in the ukiyo-e style",
        "negative_prompt": "photograph, 3d rendering, western oil painting, jpeg artifacts",
        "lora_scale": 1.0,
        "i2i_strength": 0.50,
        "i2i_guidance_scale": 8.5,
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
    style_type: Literal["cubism", "pop-art", "post-impressionism", "ukiyo-e"]
    prompt: str | None = None
    init_image: str | None = None
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
    init_payload = payload.split(",", 1)[1] if "," in payload else payload
    init_bytes = base64.b64decode(init_payload)
    init_img = Image.open(BytesIO(init_bytes))
    return ImageOps.exif_transpose(init_img).convert("RGB")


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
        training_caption = str(preset["training_caption"])
        parts: list[str] = [training_caption]
        if base_prompt:
            parts.append(base_prompt)
        return ", ".join(parts), negative_prompt

    if not base_prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt is required for text-to-image mode.",
        )

    return f"{base_prompt}, {preset['training_caption']}", negative_prompt


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
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    i2i_pipeline = StableDiffusionImg2ImgPipeline(**pipeline.components)
    i2i_pipeline = i2i_pipeline.to(device)
    i2i_pipeline.scheduler = EulerDiscreteScheduler.from_config(
        i2i_pipeline.scheduler.config,
    )
    i2i_pipeline.set_progress_bar_config(disable=True)

    assert id(pipeline.unet) == id(i2i_pipeline.unet), "UNet not shared between pipelines"

    print(
        f"Pipeline loaded. device={device} t2i_scheduler=euler i2i_scheduler=euler"
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
                    f" i2i_strength={preset['i2i_strength']}"
                    f" guidance={preset['i2i_guidance_scale']}"
                    if is_img2img
                    else f" guidance={T2I_GUIDANCE_SCALE}"
                )
            )

            pipeline.load_lora_weights(str(lora_path))

            with torch.inference_mode():
                if is_img2img:
                    result = i2i_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=init_img,
                        strength=float(preset["i2i_strength"]),
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
