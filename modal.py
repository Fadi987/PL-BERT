import modal

import preprocess

image = modal.Image.debian_slim(
    ).pip_install_from_requirements(
    "../arabic_audio_ai/requirements.txt"
    ).run_commands([
    "apt-get -y update",
    "apt-get -y install espeak-ng"
    ]).add_local_dir(
    "configs", remote_path="/root/configs")

app = modal.App(name="test", image=image)

volume = modal.Volume.from_name("pl_bert", create_if_missing=True)

@app.function(volumes={"/data/pl_bert": volume})
def preprocess_main():
    preprocess.main()

@app.local_entrypoint()
def main():
    preprocess_main.remote()