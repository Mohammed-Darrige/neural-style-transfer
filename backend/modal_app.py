from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "premium-portfolio-style-transfer"
ROOT = Path(__file__).resolve().parent


def _download_base_model() -> None:
    from diffusers import AutoencoderKL, StableDiffusionPipeline
    from transformers import BlipForConditionalGeneration, BlipProcessor

    StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .run_function(_download_base_model)
    .add_local_file(str(ROOT / "main.py"), remote_path="/root/main.py")
    .add_local_dir(
        str(ROOT.parent / "weights"),
        remote_path="/root/weights",
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=900,
    scaledown_window=420,
    min_containers=0,
    max_containers=2,
    env={
        "STYLE_WEIGHTS_ROOT": "/root/weights",
        "STYLE_INFERENCE_STEPS": "30",
        "STYLE_GUIDANCE_SCALE": "8.0",
        "CORS_ORIGINS": "http://localhost:3000,http://127.0.0.1:3000,https://darrige.tech",
    },
)
@modal.asgi_app()
def fastapi_endpoint():
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from main import app as fastapi_app

    return fastapi_app
