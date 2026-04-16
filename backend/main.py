import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Neural Style Transfer API")
pipeline: StableDiffusionPipeline | None = None
i2i_pipeline: StableDiffusionImg2ImgPipeline | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    lora_scale: float
    prompt: str
    init_image: str | None = None


@app.on_event("startup")
async def load_pipeline() -> None:
    global pipeline, i2i_pipeline
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("Loading base stable diffusion pipeline from Hugging Face...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    i2i_pipeline = StableDiffusionImg2ImgPipeline(**pipeline.components)
    i2i_pipeline = i2i_pipeline.to(device)
    i2i_pipeline.scheduler = EulerDiscreteScheduler.from_config(
        i2i_pipeline.scheduler.config
    )
    print("Base pipeline loaded to memory successfully.")


@app.post("/generate")
async def generate_style(request: StyleRequest):
    global pipeline, i2i_pipeline

    if pipeline is None or i2i_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    folder_style = request.style_type.lower()

    modal_path = (
        f"/root/weights/lora-output-{folder_style}/pytorch_lora_weights.safetensors"
    )
    local_path = (
        Path.cwd()
        / "weights"
        / f"lora-output-{folder_style}"
        / "pytorch_lora_weights.safetensors"
    )

    if os.path.exists(modal_path):
        lora_path = modal_path
    elif local_path.exists():
        lora_path = str(local_path)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"LoRA weights not found. Checked provider path ({modal_path}) and local path ({local_path}).",
        )

    print("[StyleRequest]", request.model_dump())

    loaded_lora = False
    image_buffer = BytesIO()
    style_suffixes = {
        "ukiyo-e": "a ukiyo-e style painting, high quality, authentic woodblock print",
        "cubism": "a cubism style painting, geometric shapes, abstract, high quality",
        "pop-art": "a pop art style painting, bold color fields, graphic contrast, high quality",
        "post-impressionism": "a post-impressionism style painting, expressive brushstrokes, vivid colors",
    }

    try:
        ui_scale = request.lora_scale
        applied_lora_weight = 0.6 + (ui_scale * 0.5)
        applied_strength = 0.15 + (ui_scale * 0.60)

        # Load weights on the fly
        pipeline.load_lora_weights(lora_path)
        pipeline.fuse_lora(lora_scale=applied_lora_weight)
        loaded_lora = True

        suffix = style_suffixes.get(folder_style, "")
        base_prompt = request.prompt.strip() if request.prompt else ""
        enhanced_prompt = f"{base_prompt}, {suffix}".strip(", ")

        with torch.inference_mode():
            if request.init_image:
                init_payload = (
                    request.init_image.split(",", 1)[1]
                    if "," in request.init_image
                    else request.init_image
                )
                init_bytes = base64.b64decode(init_payload)
                init_img = Image.open(BytesIO(init_bytes)).convert("RGB")

                result = i2i_pipeline(
                    prompt=enhanced_prompt,
                    image=init_img,
                    strength=applied_strength,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                )
            else:
                result = pipeline(
                    prompt=enhanced_prompt,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                )

        image = result[0][0] if isinstance(result, tuple) else result.images[0]
        image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()

        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        ) from exc
    finally:
        if loaded_lora:
            try:
                pipeline.unfuse_lora()
                pipeline.unload_lora_weights()
            except Exception:
                pass
