import modal 

import train

preprocess_image = modal.Image.debian_slim(
    ).pip_install_from_requirements(
    "../arabic_audio_ai/requirements.txt"
    ).run_commands([
    "apt-get -y update",
    "apt-get -y install espeak-ng"
    ]).add_local_dir(
    "configs", remote_path="/root/configs")

training_image = modal.Image.debian_slim(
    ).pip_install_from_requirements(
    "../arabic_audio_ai/requirements.txt"
    ).add_local_dir(
    "configs", remote_path="/root/configs")

app = modal.App(name="test", image=training_image)

volume = modal.Volume.from_name("pl_bert", create_if_missing=True)

@app.function(
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")])
def train_main():
    train.train()

@app.local_entrypoint()
def main():
    train_main.remote()