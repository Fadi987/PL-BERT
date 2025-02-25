import os
import argparse
import time
from typing import List, Tuple, Set, Any
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from pebble import ProcessPool
import phonemizer
from transformers import AutoTokenizer
import yaml

from phonemize import phonemize

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess text data for phoneme-level BERT training")
    parser.add_argument("--config_path", type=str, default="external/pl_bert/Configs/config.yml", help="Path to config file")
    parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia", help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="20231101.ar", help="Dataset split to use")
    parser.add_argument("--local_dataset_path", type=str, default=None, help="Path to local dataset (if using local data)")
    parser.add_argument("--use_local_dataset", action="store_true", help="Use local dataset instead of downloading from HuggingFace")
    parser.add_argument("--phonemizer_language", type=str, default="ar", help="Language for phonemizer")
    parser.add_argument("--tokenizer_name", type=str, default="aubmindlab/bert-base-arabertv2", help="Tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="external/pl_bert/wiki_phoneme", help="Directory to save processed shards")
    parser.add_argument("--num_shards", type=int, default=10000, help="Number of shards to split dataset into")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel workers")
    parser.add_argument("--max_try_count", type=int, default=3, help="Maximum number of retries for failed shards")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for processing a single shard (seconds)")
    return parser.parse_args()

def process_shard(args: Tuple[int, str, Dataset, Any, Any, int]) -> Tuple[int, bool]:
    """Process a single dataset shard."""
    i, root_directory, dataset, global_phonemizer, tokenizer, num_shards = args
    directory = os.path.join(root_directory, f"shard_{i}")
    
    if os.path.exists(directory):
        print(f"Shard {i} already exists!")
        return i, True
    
    try:
        print(f'Processing shard {i} ...')
        shard = dataset.shard(num_shards=num_shards, index=i)
        processed_dataset = shard.map(
            lambda t: phonemize(t['text'], global_phonemizer, tokenizer), 
            remove_columns=['text']
        )
        
        os.makedirs(directory, exist_ok=True)
        processed_dataset.save_to_disk(directory)
        print(f'Shard {i} saved to {directory}')
        return i, True
    except Exception as e:
        print(f"Error processing shard {i}: {str(e)}")
        return i, False

def get_existing_shards(root_directory: str) -> Set[int]:
    """Get the set of already processed shard IDs."""
    existing_shards = set()
    if os.path.exists(root_directory):
        for dirname in os.listdir(root_directory):
            if dirname.startswith("shard_"):
                try:
                    shard_id = int(dirname.split("_")[1])
                    existing_shards.add(shard_id)
                except ValueError:
                    continue
    return existing_shards

def process_missing_shards(
    missing_shards: List[int], 
    root_directory: str, 
    dataset: Dataset, 
    global_phonemizer: Any, 
    tokenizer: Any, 
    num_shards: int, 
    max_workers: int,
    timeout: int
) -> List[int]:
    """Process missing shards in parallel and return still missing shards."""
    if not missing_shards:
        return []
        
    print(f"Processing {len(missing_shards)} shards...")
    process_args = [
        (i, root_directory, dataset, global_phonemizer, tokenizer, num_shards) 
        for i in missing_shards
    ]
    
    results = []
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(process_shard, process_args, timeout=timeout)
        
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                results.append(result)
            except StopIteration:
                break
            except Exception as e:
                print(f"Error in worker: {str(e)}")
    
    # Identify which shards were successfully processed
    successful_shards = [shard_id for shard_id, success in results if success]
    still_missing = [i for i in missing_shards if i not in successful_shards]
    
    print(f"Completed {len(successful_shards)} shards in this attempt")
    print(f"Still missing {len(still_missing)} shards")
    
    return still_missing

def load_all_shards(root_directory: str) -> List[Dataset]:
    """Load all processed shards from disk."""
    datasets = []
    for dirname in os.listdir(root_directory):
        if dirname.startswith("shard_") and os.path.isdir(os.path.join(root_directory, dirname)):
            try:
                shard = load_from_disk(os.path.join(root_directory, dirname))
                datasets.append(shard)
                print(f"{dirname} loaded")
            except Exception as e:
                print(f"Error loading {dirname}: {str(e)}")
    return datasets

def combine_and_save_dataset(datasets: List[Dataset], output_path: str) -> None:
    """Combine all shards and save the final dataset."""
    if not datasets:
        print("Error: No shards were successfully processed")
        return
    
    final_dataset = concatenate_datasets(datasets)
    final_dataset.save_to_disk(output_path)
    print(f'Dataset saved to {output_path}')
    print(f'Total samples: {len(final_dataset)}')

def main():
    """Main function to orchestrate the preprocessing pipeline."""
    args = parse_args()
    
    # Load config
    config = yaml.safe_load(open(args.config_path))
    
    # Initialize phonemizer and tokenizer
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language=args.phonemizer_language, 
        preserve_punctuation=True, 
        with_stress=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Load dataset - either from HuggingFace or local path
    if args.use_local_dataset:
        local_path = args.local_dataset_path or config.get('local_dataset_path')
        if not local_path:
            raise ValueError("Local dataset path must be provided when using local dataset")
        print(f"Loading local dataset from {local_path}...")
        dataset = load_from_disk(local_path)
        print(f"Local dataset loaded with {len(dataset)} samples")
    else:
        print(f"Loading dataset {args.dataset_name} ({args.dataset_split}) from HuggingFace...")
        dataset = load_dataset(args.dataset_name, args.dataset_split, trust_remote_code=True)['train']
        print(f"Dataset loaded with {len(dataset)} samples")
    
    # Setup processing parameters
    root_directory = args.output_dir
    os.makedirs(root_directory, exist_ok=True)
    
    # Process all shards with retries
    all_shards = list(range(args.num_shards))
    try_count = 0
    
    while try_count < args.max_try_count:
        try_count += 1
        
        # Check for already processed shards
        existing_shards = get_existing_shards(root_directory)
        missing_shards = [i for i in all_shards if i not in existing_shards]
        
        if not missing_shards:
            print("All shards have been processed successfully!")
            break
            
        print(f"Processing attempt {try_count}/{args.max_try_count} for {len(missing_shards)} shards")
        
        # Process missing shards
        missing_shards = process_missing_shards(
            missing_shards, 
            root_directory, 
            dataset, 
            global_phonemizer, 
            tokenizer, 
            args.num_shards, 
            args.max_workers,
            args.timeout
        )
        
        if not missing_shards:
            print("All shards have been processed successfully!")
            break
            
        if try_count < args.max_try_count:
            wait_time = 10 * try_count  # Increase wait time with each retry
            print(f"Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)
    
    if missing_shards:
        print(f"Warning: {len(missing_shards)} shards could not be processed after {args.max_try_count} attempts")
        print(f"Missing shards: {missing_shards}")
    
    # Combine all shards
    print("Combining all processed shards...")
    datasets = load_all_shards(root_directory)
    
    # Save final dataset
    output_path = config.get('data_folder', os.path.join(args.output_dir, 'final_dataset'))
    combine_and_save_dataset(datasets, output_path)

if __name__ == "__main__":
    main()