import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from pebble import ProcessPool
from phonemize import phonemize
import phonemizer
from transformers import AutoTokenizer
import yaml

# Load config
config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))

# Initialize phonemizer and tokenizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='ar', preserve_punctuation=True, with_stress=True)
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

# Load dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.ar", trust_remote_code=True)['train']

# Setup processing parameters
root_directory = "./wiki_phoneme"
num_shards = 10000
max_workers = 32  # Change this to match available CPU cores

def process_shard(i):
    directory = os.path.join(root_directory, f"shard_{i}")
    if os.path.exists(directory):
        print(f"Shard {i} already exists!")
        return
    
    print(f'Processing shard {i} ...')
    shard = dataset.shard(num_shards=num_shards, index=i)
    processed_dataset = shard.map(lambda t: phonemize(t['text'], global_phonemizer, tokenizer), remove_columns=['text'])
    
    os.makedirs(directory, exist_ok=True)
    processed_dataset.save_to_disk(directory)
    print(f'Shard {i} saved to {directory}')

# Process shards in parallel
with ProcessPool(max_workers=max_workers) as pool:
    pool.map(process_shard, range(num_shards), timeout=60)

# Check for missing shards
existing_shards = set()
for dirname in os.listdir(root_directory):
    if dirname.startswith("shard_"):
        try:
            shard_id = int(dirname.split("_")[1])
            existing_shards.add(shard_id)
        except ValueError:
            continue

missing_shards = [i for i in range(num_shards) if i not in existing_shards]
print(f"Found {len(missing_shards)} missing shards:")
print(missing_shards)

# Combine all shards
datasets = []
for dirname in os.listdir(root_directory):
    if os.path.isdir(os.path.join(root_directory, dirname)):
        try:
            shard = load_from_disk(os.path.join(root_directory, dirname))
            datasets.append(shard)
            print(f"{dirname} loaded")
        except:
            continue

# Save final dataset
dataset = concatenate_datasets(datasets)
dataset.save_to_disk(config['data_folder'])
print(f'Dataset saved to {config["data_folder"]}')