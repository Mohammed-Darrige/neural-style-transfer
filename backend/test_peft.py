import modal


app = modal.App("style-backend-peft-test")


@app.function(
    image=modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
)
def test() -> None:
    try:
        import peft

        print("PEFT VERSION:", peft.__version__)
    except Exception as exc:  # pragma: no cover
        print("ERR:", exc)


if __name__ == "__main__":
    with app.run():
        test.remote()
