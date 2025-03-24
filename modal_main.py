import modal 

import train

restart_tracker_dict = modal.Dict.from_name(
    "restart-tracker", create_if_missing=True
)

def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count

# Images
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
    _ = track_restarts(restart_tracker_dict)
    train.train({"config_path": "/root/configs/config.yml", "run_name": "default"})

@app.local_entrypoint()
def main():
    train_main.remote()