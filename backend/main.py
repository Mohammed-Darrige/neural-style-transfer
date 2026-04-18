import base64
import os
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionPipeline,
)
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import BlipForConditionalGeneration, BlipProcessor

app = FastAPI(title="Neural Style Transfer API")
pipeline: StableDiffusionPipeline | None = None
i2i_pipeline: StableDiffusionControlNetImg2ImgPipeline | None = None
caption_processor: BlipProcessor | None = None
caption_model: BlipForConditionalGeneration | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"
supports_multi_adapter = False
loaded_adapters: set[str] = set()
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
I2I_INFERENCE_STEPS = int(os.getenv("STYLE_I2I_INFERENCE_STEPS", "35"))
I2I_MIN_DIM = int(os.getenv("STYLE_I2I_MIN_DIM", str(IMAGE_SIZE)))
I2I_MAX_DIM = int(os.getenv("STYLE_I2I_MAX_DIM", "768"))
CAPTION_MAX_TOKENS = int(os.getenv("STYLE_CAPTION_MAX_TOKENS", "28"))
CAPTION_MODEL_ID = os.getenv(
    "STYLE_CAPTION_MODEL_ID",
    "Salesforce/blip-image-captioning-base",
)
DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "STYLE_NEGATIVE_PROMPT",
    "low quality, worst quality, blurry, muddy colors, dark exposure, underexposed, distorted, deformed, duplicate subjects, extra limbs, cropped, out of frame, watermark, text, signature",
)

# ──────────────────────────────────────────────────────────────────────────────
# Style presets — redesigned for ControlNet-backed img2img.
#
# Key changes vs. previous version:
#   • i2i_strength raised to 0.72-0.82  (was 0.46-0.56)
#     ControlNet now anchors structure, so higher strength is safe and necessary
#     for visible style transfer.
#   • lora_scale raised to 0.88-0.95  (was 0.80-0.86)
#   • Added per-style controlnet_conditioning_scale and i2i_guidance_scale.
#   • prompt_prefix added — for img2img the style description LEADS the prompt,
#     because the init image already carries the composition/identity.
#   • identity_prompt removed from img2img prompts entirely — ControlNet handles
#     structure preservation now, so prompt-based identity anchoring is
#     counter-productive (it fights the style).
# ──────────────────────────────────────────────────────────────────────────────

STYLE_PRESETS: dict[str, dict[str, object]] = {
    "cubism": {
        "prompt_prefix": "cubist painting, geometric abstraction, angular faceted planes, bold outlines",
        "prompt_suffix": "an artwork in the cubism style, geometric abstraction, angular composition",
        "negative_prompt": "photorealistic, smooth skin, soft focus, photograph, watercolor wash, muddy geometry, broken anatomy",
        "lora_scale": 0.92,
        "i2i_strength": 0.78,
        "controlnet_scale": 0.85,
        "i2i_guidance_scale": 8.0,
        "canny_low": 60,
        "canny_high": 160,
    },
    "pop-art": {
        "prompt_prefix": "pop art painting, bold flat color fields, high graphic contrast, comic halftone dots",
        "prompt_suffix": "an artwork in the pop art style, bold color fields, graphic contrast",
        "negative_prompt": "photorealistic, muted colors, soft gradients, sepia, grayscale, photograph, washed out",
        "lora_scale": 0.90,
        "i2i_strength": 0.75,
        "controlnet_scale": 0.88,
        "i2i_guidance_scale": 7.5,
        "canny_low": 70,
        "canny_high": 170,
    },
    "post-impressionism": {
        "prompt_prefix": "post-impressionist painting, expressive visible brushstrokes, vivid color rhythm, painterly texture",
        "prompt_suffix": "an artwork in the post-impressionism style, expressive brushwork, vivid color rhythm",
        "negative_prompt": "photorealistic, flat digital shading, smooth blending, photograph, muddy paint, weak brush texture",
        "lora_scale": 0.93,
        "i2i_strength": 0.80,
        "controlnet_scale": 0.82,
        "i2i_guidance_scale": 7.5,
        "canny_low": 50,
        "canny_high": 150,
    },
    "ukiyo-e": {
        "prompt_prefix": "ukiyo-e woodblock print, flat color areas, elegant black contour lines, Japanese art",
        "prompt_suffix": "an artwork in the ukiyo-e style, woodblock print aesthetics, elegant contour lines",
        "negative_prompt": "photorealistic, 3d rendering, soft gradients, photograph, western oil painting, muddy shading",
        "lora_scale": 0.90,
        "i2i_strength": 0.76,
        "controlnet_scale": 0.90,
        "i2i_guidance_scale": 7.5,
        "canny_low": 80,
        "canny_high": 180,
    },
}

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
PERSON_TOKENS = {
    "man",
    "woman",
    "person",
    "boy",
    "girl",
    "male",
    "female",
    "portrait",
    "face",
    "selfie",
    "bride",
    "groom",
    "people",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PRELOAD_ADAPTERS = _env_bool("STYLE_PRELOAD_ADAPTERS", True)

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


def _ensure_adapter_loaded(style: str, lora_path: Path) -> bool:
    global loaded_adapters

    if style in loaded_adapters:
        return True

    # Both pipelines share the same model components, so loading once is enough.
    try:
        pipeline.load_lora_weights(str(lora_path), adapter_name=style)
    except TypeError:
        return False

    loaded_adapters.add(style)
    return True


def _activate_adapter(style: str, weight: float) -> None:
    """Activate adapter using set_adapters (reliable) instead of cross_attention_kwargs."""
    pipeline.set_adapters(adapter_names=[style], adapter_weights=[weight])


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


def _extract_canny_edges(
    image: Image.Image,
    low_threshold: int = 80,
    high_threshold: int = 180,
) -> Image.Image:
    """Extract Canny edges from a PIL image for ControlNet conditioning.

    Returns a 3-channel RGB image with white edges on a black background,
    matching the format expected by the Canny ControlNet model.
    """
    img_np = np.array(image)
    # Convert to grayscale for edge detection
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # ControlNet expects 3-channel image
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)


def _clean_caption(text: str) -> str:
    cleaned = " ".join(text.strip().lower().split())
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _caption_init_image(image: Image.Image) -> str:
    if caption_processor is None or caption_model is None:
        return ""

    caption_inputs = caption_processor(images=image, return_tensors="pt")
    pixel_values = caption_inputs["pixel_values"]
    device_inputs = {"pixel_values": pixel_values.to("cpu")}

    with torch.inference_mode():
        generated = caption_model.generate(
            **device_inputs,
            max_new_tokens=CAPTION_MAX_TOKENS,
            num_beams=4,
        )

    return _clean_caption(
        caption_processor.decode(generated[0], skip_special_tokens=True)
    )


def _compose_prompt(
    style: str,
    prompt: str | None,
    is_img2img: bool,
    source_caption: str | None = None,
) -> tuple[str, str]:
    """Build the final prompt and negative prompt.

    For img2img (ControlNet-backed):
      - Style description LEADS the prompt — the init image already carries
        composition and identity, so the text prompt should maximize style signal.
      - Caption is included but subordinated to the style prefix.
      - No "preserve identity" instructions — ControlNet handles that.

    For text-to-image:
      - User prompt leads, style suffix appended.
    """
    preset = STYLE_PRESETS[style]
    negative_prompt = f"{DEFAULT_NEGATIVE_PROMPT}, {preset['negative_prompt']}"

    base_prompt = (prompt or "").strip()

    if is_img2img:
        prefix = str(preset["prompt_prefix"])
        suffix = str(preset["prompt_suffix"])

        prompt_parts: list[str] = [prefix]

        # Insert caption as subordinate context (not leading)
        if source_caption:
            prompt_parts.append(f"depicting {source_caption}")

        # User-supplied extra direction
        if base_prompt:
            prompt_parts.append(base_prompt)

        prompt_parts.append(suffix)
        return ", ".join(part for part in prompt_parts if part), negative_prompt

    # Text-to-image path — unchanged from original
    suffix = str(preset["prompt_suffix"])
    if not base_prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt is required for text-to-image mode.",
        )

    return f"{base_prompt}, {suffix}", negative_prompt


def _is_near_black_image(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    hist = grayscale.histogram()
    total = sum(hist)
    if total <= 0:
        return True

    dark_pixels = sum(hist[:10])
    dark_ratio = dark_pixels / total
    mean_luma = sum(index * count for index, count in enumerate(hist)) / total

    return dark_ratio > 0.96 or mean_luma < 12


@app.on_event("startup")
async def load_pipeline() -> None:
    global pipeline, i2i_pipeline, caption_processor, caption_model
    global supports_multi_adapter, loaded_adapters
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("Loading base stable diffusion pipeline from Hugging Face...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading caption model: {CAPTION_MODEL_ID}")
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(
        "cpu"
    )
    caption_model.eval()
    
    # FIX for SD 1.5 "Black Image" VAE NaN issue in fp16
    # We load the fp16-safe MSE VAE instead of upcasting to float32
    print("Loading fp16-safe VAE (sd-vae-ft-mse)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch_dtype
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

    # ControlNet for structural preservation during img2img
    print("Loading ControlNet (Canny) for img2img structural anchoring...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch_dtype,
    ).to(device)

    # img2img pipeline now uses ControlNet for structure preservation.
    # This lets us use much higher denoising strength (0.72-0.82) while keeping
    # the composition/identity locked via Canny edges.
    i2i_pipeline = StableDiffusionControlNetImg2ImgPipeline(
        **pipeline.components,
        controlnet=controlnet,
    )
    i2i_pipeline = i2i_pipeline.to(device)
    i2i_pipeline.scheduler = EulerDiscreteScheduler.from_config(
        i2i_pipeline.scheduler.config
    )
    i2i_pipeline.set_progress_bar_config(disable=True)

    supports_multi_adapter = False

    if PRELOAD_ADAPTERS and supports_multi_adapter:
        for style in SUPPORTED_STYLES:
            lora_path = _resolve_lora_path(style)
            try:
                pipeline.load_lora_weights(str(lora_path), adapter_name=style)
                loaded_adapters.add(style)
            except TypeError:
                supports_multi_adapter = False
                loaded_adapters.clear()
                break

    print(
        f"Base pipeline loaded. device={device} steps={INFERENCE_STEPS} multi_adapter={supports_multi_adapter} preloaded={len(loaded_adapters)} controlnet=canny"
    )


@app.post("/generate")
async def generate_style(request: StyleRequest):
    global pipeline, i2i_pipeline, loaded_adapters, supports_multi_adapter

    if pipeline is None or i2i_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    folder_style = request.style_type.lower()
    lora_path = _resolve_lora_path(folder_style)
    preset = STYLE_PRESETS[folder_style]
    is_img2img = request.init_image is not None

    print(
        "[StyleRequest]",
        {
            "style_type": request.style_type,
            "prompt": request.prompt,
            "has_init_image": is_img2img,
        },
    )

    loaded_lora = False
    image_buffer = BytesIO()
    init_source_image: Image.Image | None = None
    init_img: Image.Image | None = None
    canny_image: Image.Image | None = None
    source_caption = ""

    try:
        with INFERENCE_LOCK:
            if is_img2img:
                init_source_image = _decode_init_image(request.init_image or "")
                source_caption = _caption_init_image(init_source_image)
                init_img = _prepare_init_image(init_source_image)

                # Extract Canny edges for ControlNet structural conditioning
                canny_low = int(preset.get("canny_low", 80))
                canny_high = int(preset.get("canny_high", 180))
                canny_image = _extract_canny_edges(init_img, canny_low, canny_high)

            applied_lora_weight = float(preset["lora_scale"])
            enhanced_prompt, negative_prompt = _compose_prompt(
                folder_style,
                request.prompt,
                is_img2img,
                source_caption=source_caption,
            )

            if source_caption:
                print(f"[SourceCaption] {source_caption}")
            print(f"[ResolvedPrompt] {enhanced_prompt}")
            print(
                f"[Params] lora_scale={applied_lora_weight}"
                + (
                    f" i2i_strength={preset['i2i_strength']}"
                    f" controlnet_scale={preset['controlnet_scale']}"
                    f" guidance={preset['i2i_guidance_scale']}"
                    if is_img2img
                    else f" guidance={T2I_GUIDANCE_SCALE}"
                )
            )

            if supports_multi_adapter:
                if _ensure_adapter_loaded(folder_style, lora_path):
                    _activate_adapter(folder_style, applied_lora_weight)
                else:
                    supports_multi_adapter = False
                    loaded_adapters.clear()
                    pipeline.load_lora_weights(str(lora_path))
                    loaded_lora = True
            else:
                pipeline.load_lora_weights(str(lora_path))
                loaded_lora = True

            # Use set_adapters for reliable LoRA weight control.
            # cross_attention_kwargs is unreliable in modern diffusers.
            if not supports_multi_adapter:
                try:
                    pipeline.set_adapters(
                        adapter_names=["default"],
                        adapter_weights=[applied_lora_weight],
                    )
                except Exception:
                    # Fallback: some diffusers versions don't support
                    # set_adapters with "default" name after load_lora_weights
                    # without adapter_name — fall through to cross_attention_kwargs
                    pass

            with torch.inference_mode():
                if is_img2img:
                    result = i2i_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=init_img,
                        control_image=canny_image,
                        strength=float(preset["i2i_strength"]),
                        controlnet_conditioning_scale=float(
                            preset["controlnet_scale"]
                        ),
                        guidance_scale=float(preset["i2i_guidance_scale"]),
                        num_inference_steps=I2I_INFERENCE_STEPS,
                    )
                else:
                    result = pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=T2I_GUIDANCE_SCALE,
                        num_inference_steps=INFERENCE_STEPS,
                        width=IMAGE_SIZE,
                        height=IMAGE_SIZE,
                        cross_attention_kwargs=(
                            None
                            if supports_multi_adapter
                            else {"scale": applied_lora_weight}
                        ),
                    )

            image = result[0][0] if isinstance(result, tuple) else result.images[0]

            if _is_near_black_image(image):
                if is_img2img:
                    fallback_strength = max(0.60, float(preset["i2i_strength"]) - 0.10)
                    fallback_guidance = max(6.0, float(preset["i2i_guidance_scale"]) - 0.5)
                    fallback_cn_scale = min(1.0, float(preset["controlnet_scale"]) + 0.05)
                    print(
                        f"[Fallback] Near-black detected. Retrying with strength={fallback_strength}"
                        f" guidance={fallback_guidance} cn_scale={fallback_cn_scale}"
                    )
                    with torch.inference_mode():
                        fallback_result = i2i_pipeline(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            image=init_img,
                            control_image=canny_image,
                            strength=fallback_strength,
                            controlnet_conditioning_scale=fallback_cn_scale,
                            guidance_scale=fallback_guidance,
                            num_inference_steps=I2I_INFERENCE_STEPS,
                        )
                    candidate = (
                        fallback_result[0][0]
                        if isinstance(fallback_result, tuple)
                        else fallback_result.images[0]
                    )
                    if not _is_near_black_image(candidate):
                        image = candidate
                else:
                    fallback_guidance = max(5.5, T2I_GUIDANCE_SCALE - 1.0)
                    with torch.inference_mode():
                        fallback_result = pipeline(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=fallback_guidance,
                            num_inference_steps=INFERENCE_STEPS,
                            width=IMAGE_SIZE,
                            height=IMAGE_SIZE,
                            cross_attention_kwargs=(
                                None
                                if supports_multi_adapter
                                else {"scale": applied_lora_weight}
                            ),
                        )
                    candidate = (
                        fallback_result[0][0]
                        if isinstance(fallback_result, tuple)
                        else fallback_result.images[0]
                    )
                    if not _is_near_black_image(candidate):
                        image = candidate

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
        if loaded_lora:
            try:
                pipeline.unload_lora_weights()
            except Exception:
                pass
