import modal

def download_model():
    from diffusers import StableDiffusionPipeline
    print("Downloading StableDiffusion v1.5 onto Modal Image...")
    StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .run_function(download_model)
    .add_local_dir("weights", remote_path="/root/weights")
    .add_local_file("main.py", remote_path="/root/main.py")
)

app = modal.App("personal-lab-backend")

@app.function(image=image, gpu="L4")
@modal.asgi_app()
def fastapi_endpoint():
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from main import app as fastapi_app
    return fastapi_app
