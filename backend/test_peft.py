import modal

app = modal.App('test')

@app.function(image=modal.Image.debian_slim().pip_install_from_requirements('requirements.txt'))
def test():
    try:
        import peft
        print('PEFT VERSION:', peft.__version__)
    except Exception as e:
        print('ERR:', e)

if __name__ == '__main__':
    with app.run():
        test.remote()
