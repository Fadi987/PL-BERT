import os
import argparse
import time
import shutil
from typing import List, Tuple, Set, Any, Callable
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from pebble import ProcessPool
import phonemizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml
import numpy as np

from char_indexer import PUNCTUATION
from dataloader import TruncatedTextDataset
from text_normalize import convert_numbers_to_arabic_words, filter_non_arabic_words, remove_diacritics, separate_words_and_punctuation, clean_text
from util_models import CattTashkeel

def standardize_text(text):
    """Perform text cleaning operations without phonemization.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    text = convert_numbers_to_arabic_words(text)
    text = filter_non_arabic_words(text)
    text = clean_text(text)
    return text

def separate_text_into_segments(tokens):
    """Separate tokens into text segments and punctuation.
    
    Args:
        tokens: List of tokens to separate
        
    Returns:
        Tuple containing:
            - segments: List of text segments
            - punctuations: List of punctuation tokens
            - segment_indices: List of indices where segments end
    """
    segments = []
    current_segment = []
    punctuations = []
    segment_indices = []  # To track where punctuations should be inserted
    
    for i, token in enumerate(tokens):
        if token in PUNCTUATION:
            if current_segment:
                segments.append(" ".join(current_segment))
                segment_indices.append(i)
                current_segment = []
            punctuations.append(token)
        else:
            current_segment.append(token)
    
    # Add the last segment if it exists
    if current_segment:
        segments.append(" ".join(current_segment))
        segment_indices.append(len(tokens))
        
    return segments, punctuations, segment_indices

def phonemize_text(text, phonemizer_instance):
    """Perform phonemization without diacritization.
    
    Args:
        text: Input text to phonemize
        phonemizer_instance: Phonemizer instance
        
    Returns:
        List of phonemes
    """
    # Extract punctuation and text segments
    tokens = separate_words_and_punctuation(text)
    
    # Group non-punctuation tokens into segments
    segments, punctuations, segment_indices = separate_text_into_segments(tokens)
    
    # Phonemize each segment and split into words
    phonemized_segments = []
    for segment in segments:
        phonemized_segment = phonemizer_instance.phonemize([segment], strip=True)[0]
        # Split the phonemized segment into words
        phonemized_words = phonemized_segment.split()
        phonemized_segments.extend(phonemized_words)
    
    # Reconstruct the final phonemes list with punctuation in the correct positions
    phonemes = []
    seg_idx = 0
    punct_idx = 0
    
    for i in range(len(tokens)):
        if i in segment_indices:
            # We've reached the end of a segment, add punctuation
            if punct_idx < len(punctuations):
                phonemes.append(punctuations[punct_idx])
                punct_idx += 1
        else:
            # Add the next phonemized word from the current segment
            if seg_idx < len(phonemized_segments):
                phonemes.append(phonemized_segments[seg_idx])
                seg_idx += 1
                
    return phonemes

def diacritize_text(text, diacritizer=None):
    """Convert text to phonemes using diacritization and word segmentation.
    
    Args:
        text: Input text to phonemize
        global_phonemizer: Phonemizer instance
        diacritizer: Optional diacritizer for Arabic text
        
    Returns:
        List of phonemes
    """
    # Extract punctuation and text segments
    tokens = separate_words_and_punctuation(text)
    
    # Group non-punctuation tokens into segments
    segments, punctuations, segment_indices = separate_text_into_segments(tokens)
    
    # Diacritize the segments if diacritizer is provided
    if diacritizer is not None:
        diacritized_segments = diacritizer.do_tashkeel_batch(segments, batch_size=16, verbose=False)
    else:
        diacritized_segments = segments

    # Split segments into tokens
    diacritized_tokens = []
    for segment in diacritized_segments:
        diacritized_tokens.extend(segment.split())

    # Reconstruct the final diacritized text with punctuation in the correct positions
    diacritized_text = ""
    token_idx = 0
    punct_idx = 0
    
    for i in range(len(tokens)):
        if i in segment_indices:
            # We've reached the end of a segment, add punctuation
            if punct_idx < len(punctuations):
                diacritized_text += punctuations[punct_idx]
                punct_idx += 1
        else:
            # Add the next diacritized token
            if token_idx < len(diacritized_tokens):
                # Add space before word unless it's the first word
                if diacritized_text and not diacritized_text.endswith(" "):
                    diacritized_text += " "
                diacritized_text += diacritized_tokens[token_idx]
                token_idx += 1
    
    # Return the reconstructed diacritized text
    return diacritized_text

def phonemize_with_tokenizer(text, global_phonemizer, tokenizer):
    """Convert text to phonemes using tokenizer.
    
    Args:
        text: Input text to phonemize
        global_phonemizer: Phonemizer instance
        tokenizer: Tokenizer instance
        
    Returns:
        Dictionary with phonemes and token_ids
    """
    tokens = tokenizer.tokenize(text)
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    phonemes = [global_phonemizer.phonemize([token.replace("#", "")], strip=True)[0] 
               if token not in PUNCTUATION else token for token in tokens]
    
    return {'phonemes': phonemes, 'token_ids': token_ids}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean/phonemize/diacritize text data for PL-BERT training")
    parser.add_argument("--config_path", type=str, default="external/pl_bert/configs/config.yml", help="Path to config file")
    parser.add_argument("--local_dataset_path", type=str, default=None, help="Path to local dataset (if using local data)")
    parser.add_argument("--use_local_dataset", action="store_true", help="Use local dataset instead of downloading from HuggingFace")
    return parser.parse_args()

def process_shard(args: Tuple[int, str, Dataset, int, Callable, Any]) -> Tuple[int, bool]:
    """Process a single dataset shard.
    
    Args:
        args: Tuple containing:
            - i: Shard index
            - root_directory: Directory to save shards
            - dataset: Dataset to process
            - num_shards: Total number of shards
            - process_fn: Function to apply to each sample
            - process_args: Additional arguments for process_fn
    """
    i, root_directory, dataset, num_shards, process_fn, process_args = args
    directory = os.path.join(root_directory, f"shard_{i}")
    
    if os.path.exists(directory):
        print(f"Shard {i} already exists!")
        return i, True
    
    try:
        print(f'Processing shard {i} ...')
        shard = dataset.shard(num_shards=num_shards, index=i)
        
        # Process the shard with the provided function
        if process_args:
            processed_dataset = shard.map(
                lambda t: process_fn(t, process_args),
            )
        else:
            processed_dataset = shard.map(
                lambda t: {'text': process_fn(t['text'])},
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
    num_shards: int, 
    max_workers: int,
    timeout: int,
    process_fn: Callable,
    process_args: Any = None
) -> List[int]:
    """Process missing shards in parallel and return still missing shards."""
    if not missing_shards:
        return []
        
    print(f"Processing {len(missing_shards)} shards...")
    process_args_list = [
        (i, root_directory, dataset, num_shards, process_fn, process_args) 
        for i in missing_shards
    ]
    
    results = []
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(process_shard, process_args_list, timeout=timeout)
        
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

def cleanup_shards(root_directory: str) -> None:
    """Remove all shard directories to clean up disk space."""
    print("Cleaning up shard directories...")
    count = 0
    for dirname in os.listdir(root_directory):
        if dirname.startswith("shard_") and os.path.isdir(os.path.join(root_directory, dirname)):
            try:
                shutil.rmtree(os.path.join(root_directory, dirname))
                count += 1
            except Exception as e:
                print(f"Error removing {dirname}: {str(e)}")
    print(f"Removed {count} shard directories")

def process_dataset(dataset, root_directory, process_fn, process_args=None, output_dir=None, max_workers=4, timeout=3600, max_try_count=3, num_shards=10000):
    """Generic function to process a dataset with the given function.
    
    Args:
        dataset: Dataset to process
        root_directory: Directory to save shards
        process_fn: Function to apply to each sample
        process_args: Additional arguments for process_fn
        output_dir: Directory name for the output dataset
        max_workers: Maximum number of parallel workers
        timeout: Timeout for each worker
        max_try_count: Maximum number of retry attempts
        
    Returns:
        Path to the processed dataset
    """
    os.makedirs(root_directory, exist_ok=True)
    
    # Process all shards with retries
    all_shards = list(range(num_shards))
    try_count = 0
    
    while try_count < max_try_count:
        try_count += 1
        
        # Check for already processed shards
        existing_shards = get_existing_shards(root_directory)
        missing_shards = [i for i in all_shards if i not in existing_shards]
        
        if not missing_shards:
            print("All shards have been processed successfully!")
            break
            
        print(f"Processing attempt {try_count}/{max_try_count} for {len(missing_shards)} shards")
        
        # Process missing shards
        missing_shards = process_missing_shards(
            missing_shards, 
            root_directory, 
            dataset, 
            num_shards, 
            max_workers,
            timeout,
            process_fn,
            process_args
        )
        
        if not missing_shards:
            print("All shards have been processed successfully!")
            break
            
        if try_count < max_try_count:
            wait_time = 10 * try_count  # Increase wait time with each retry
            print(f"Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)
    
    if missing_shards:
        print(f"Warning: {len(missing_shards)} shards could not be processed after {max_try_count} attempts")
        print(f"Missing shards: {missing_shards}")
    
    # Combine all shards
    print("Combining all processed shards...")
    datasets = load_all_shards(root_directory)
    
    # Save final dataset
    output_path = os.path.join(root_directory, output_dir) if output_dir else root_directory
    combine_and_save_dataset(datasets, output_path)
    
    # Clean up shard directories to free disk space
    cleanup_shards(root_directory)

    return output_path

def phonemize_wrapper(sample, phonemizer_instance):
    """Wrapper function to phonemize a sample using the provided phonemizer.
    
    Args:
        sample: Sample containing text to phonemize
        phonemizer_instance: Phonemizer instance to use
        
    Returns:
        Dictionary with original text and phonemized text
    """
    return {'text': sample['text'], 'phonemes': phonemize_text(sample['text'], phonemizer_instance)}

def main_clean():
    """Main function to orchestrate the text cleaning pipeline."""
    args = parse_args()
    
    # Load config
    config = yaml.safe_load(open(args.config_path))
    preprocess_params = config.get('preprocess_params', {})
    
    # Load dataset - either from HuggingFace or local path
    if args.use_local_dataset:
        local_path = args.local_dataset_path or config.get('local_dataset_path')
        if not local_path:
            raise ValueError("Local dataset path must be provided when using local dataset")
        print(f"Loading local dataset from {local_path}...")
        dataset = load_from_disk(local_path)
        print(f"Local dataset loaded with {len(dataset)} samples")
    else:
        hf_dataset_name = preprocess_params.get('hf_dataset_name')
        hf_dataset_split = preprocess_params.get('hf_dataset_split')
        print(f"Loading dataset {hf_dataset_name} ({hf_dataset_split}) from HuggingFace...")
        dataset = load_dataset(hf_dataset_name, hf_dataset_split, trust_remote_code=True)['train']
        print(f"Dataset loaded with {len(dataset)} samples")
    
    # Setup processing parameters
    root_directory = preprocess_params.get('preprocess_dir')
    
    # Process the dataset with text cleaning
    output_path = process_dataset(
        dataset=dataset,
        root_directory=root_directory,
        process_fn=standardize_text,
        output_dir=preprocess_params.get('cleaned_output_dir'),
        max_workers=preprocess_params.get('max_workers'),
        timeout=preprocess_params.get('timeout'),
        max_try_count=preprocess_params.get('max_try_count'),
        num_shards=preprocess_params.get('num_shards')
    )
    
    return output_path

def main_phonemize(dataset_path, output_dir=None):
    """Main function to orchestrate the phonemization pipeline."""
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Initialize phonemizer
    print("Initializing phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(language='ar', preserve_punctuation=True, with_stress=True)
    
    # Setup processing parameters
    root_directory = os.path.dirname(dataset_path)
    
    # Process the dataset with phonemization
    if output_dir is None:
        output_dir = os.path.basename(dataset_path).replace('cleaned', 'phonemized')
    
    output_path = process_dataset(
        dataset=dataset,
        root_directory=root_directory,
        process_fn=phonemize_wrapper,
        process_args=global_phonemizer,
        output_dir=output_dir,
        max_workers=4,
        timeout=3600,
        max_try_count=3
    )
    
    return output_path

def main_diacritize(dataset_path, output_dir=None, sample_size=200000):
    """Main function to orchestrate the diacritization pipeline.
    
    Args:
        dataset_path: Path to the dataset to diacritize
        output_dir: Optional output directory name
        sample_size: Number of samples to use (default: 200k)
        
    Returns:
        Path to the processed dataset
    """
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Create a TruncatedTextDataset
    print("Creating truncated dataset...")
    truncated_dataset = TruncatedTextDataset(dataset, max_seq_length=512)
    
    # Sample from the dataset
    print(f"Sampling {sample_size} examples from dataset...")
    if len(truncated_dataset) > sample_size:
        # Generate random indices without replacement
        indices = np.random.choice(len(truncated_dataset), size=sample_size, replace=False)
        # Create a new dataset with only the sampled examples
        sampled_dataset = Dataset.from_dict({
            'id': [truncated_dataset[int(idx)]['id'] for idx in indices],
            'url': [truncated_dataset[int(idx)]['url'] for idx in indices],
            'title': [truncated_dataset[int(idx)]['title'] for idx in indices],
            'text': [truncated_dataset[int(idx)]['text'] for idx in indices]
        })
    else:
        print(f"Warning: Requested sample size {sample_size} is larger than dataset size {len(truncated_dataset)}. Using full dataset.")
        sampled_dataset = Dataset.from_dict({
            'id': [truncated_dataset[int(idx)]['id'] for idx in range(len(truncated_dataset))],
            'url': [truncated_dataset[int(idx)]['url'] for idx in range(len(truncated_dataset))],
            'title': [truncated_dataset[int(idx)]['title'] for idx in range(len(truncated_dataset))],
            'text': [truncated_dataset[int(idx)]['text'] for idx in range(len(truncated_dataset))]
        })
    
    # Initialize diacritizer
    print("Initializing diacritizer...")
    diacritizer = CattTashkeel()
    
    # Setup output path
    root_directory = os.path.dirname(dataset_path)
    if output_dir is None:
        output_dir = os.path.basename(dataset_path).replace('cleaned', 'diacritized')
    output_path = os.path.join(root_directory, output_dir)
    
    # Process the dataset sequentially with tqdm
    print("Diacritizing texts sequentially...")
    diacritized_texts = []
    
    for text in tqdm(sampled_dataset['text'], desc="Diacritizing"):
        # remove diacritics before diacritizing to replicate CATT Tashkeel model conditioning
        diacritized_text = diacritize_text(remove_diacritics(text), diacritizer)
        diacritized_texts.append(diacritized_text)
    
    # Create and save the processed dataset
    processed_dataset = Dataset.from_dict({
        'id': sampled_dataset['id'],
        'url': sampled_dataset['url'],
        'title': sampled_dataset['title'],
        'text': sampled_dataset['text'],
        'diacritized_text': diacritized_texts
    })
    
    os.makedirs(output_path, exist_ok=True)
    processed_dataset.save_to_disk(output_path)
    print(f'Dataset saved to {output_path}')
    print(f'Total samples: {len(processed_dataset)}')
    
    return output_path


if __name__ == "__main__":
    # output_path = main_clean()
    # output_path = main_phonemize(output_path)
    output_path = '/root/notebooks/voiceAI/arabic_audio_ai_fadi/data/pl_bert/wikipedia_20231101.ar.cleaned'
    output_path = main_diacritize(output_path)
    # create_expanded_dataset(output_path, num_epochs=2)
    # output_path = '/root/notebooks/voiceAI/arabic_audio_ai_fadi/data/pl_bert/wikipedia_20231101.ar.expanded'
    # phonemize_and_diacritize_dataset(output_path)