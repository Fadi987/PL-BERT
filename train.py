# Standard library imports
import os
import shutil
import yaml
import argparse
from collections import deque

# Third-party imports
import torch
from torch import nn
from datasets import load_from_disk
import wandb
import numpy as np

# Accelerate imports
from accelerate import Accelerator, DistributedDataParallelKwargs

# Transformers imports
from transformers import AlbertConfig, AlbertModel, AutoTokenizer
from torch.optim import AdamW

# Local imports
from dataloader import build_dataloader
from model import MultiTaskModel
from char_indexer import symbols

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train phoneme-level BERT model")
    parser.add_argument("--config_path", type=str, default="external/pl_bert/configs/config.yml", help="Path to config file")
    parser.add_argument("--run_name", type=str, default="default", help="Name of the run for organizing checkpoints")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint")
    return parser.parse_args()

def length_to_mask(lengths):
    batch_size = lengths.size(0)
    max_len = int(lengths.max().item())

    # Create position indices tensor: [[0,1,2,...], [0,1,2,...], ...]
    positions = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)

    # Use position+1 > length comparison as in the original
    mask = (positions + 1) > lengths.unsqueeze(1)
    
    return mask

def find_latest_checkpoint(log_dir):
    """
    Find the latest checkpoint in the specified directory.
    
    Args:
        log_dir: Directory containing checkpoint files
        
    Returns:
        tuple: (load_checkpoint, last_iteration)
            - load_checkpoint: Boolean indicating if a checkpoint was found
            - last_iteration: The iteration number of the latest checkpoint (0 if none found)
    """
    is_checkpoint_found = False
    last_iter = 0
    try:
        ckpts = []
        for f in os.listdir(log_dir):
            if f.startswith("step_") and os.path.isfile(os.path.join(log_dir, f)):
                try:
                    iter_num = int(f.split('_')[-1].split('.')[0])
                    ckpts.append(iter_num)
                except ValueError:
                    # Skip files with invalid format
                    continue
        
        if ckpts:  # Only proceed if we found valid checkpoint files
            last_iter = max(ckpts)
            is_checkpoint_found = True
    except Exception as e:
        print(f"Error finding checkpoints: {e}")
        last_iter = 0
        is_checkpoint_found = False
    
    return is_checkpoint_found, last_iter

def load_checkpoint(model, optimizer, log_dir, last_iter, accelerator):
    """
    Load model and optimizer state from a checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        log_dir: Directory containing checkpoint files
        last_iter: The iteration number of the checkpoint to load
        accelerator: Accelerator instance for distributed training
        
    Returns:
        tuple: (model, optimizer) with loaded state
    """
    checkpoint_path = os.path.join(log_dir, f"step_{last_iter}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    
    # Remove 'module.' prefix from keys if present
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    accelerator.print(f'Checkpoint {last_iter} loaded.')
    
    return model, optimizer

def calculate_token_loss(token_pred, token_ids, input_lengths, criterion):
    """
    Calculate the token prediction loss.
    
    Args:
        token_pred: Predicted token logits
        token_ids: Ground truth token IDs
        input_lengths: Length of each input sequence
        criterion: Loss function
        
    Returns:
        float: Average token loss
    """
    loss_token = 0
    for _s2s_pred, _text_input, _text_length in zip(token_pred, token_ids, input_lengths):
        loss_token += criterion(_s2s_pred[:_text_length], 
                                _text_input[:_text_length])
    loss_token /= token_ids.size(0)
    return loss_token

def calculate_phoneme_loss(phoneme_pred, phoneme_labels, input_lengths, masked_indices, criterion):
    """
    Calculate the phoneme prediction loss.
    
    Args:
        phoneme_pred: Predicted phoneme logits
        phoneme_labels: Ground truth phoneme labels
        input_lengths: Length of each input sequence
        masked_indices: Indices of masked phonemes
        criterion: Loss function
        
    Returns:
        float: Average phoneme loss
    """
    loss_phoneme = 0
    count = 0
    for _s2s_pred, _text_input, _text_length, _masked_indices in zip(phoneme_pred, phoneme_labels, input_lengths, masked_indices):
        if len(_masked_indices) > 0:
            loss_tmp = criterion(_s2s_pred[:_text_length][_masked_indices], 
                                 _text_input[:_text_length][_masked_indices])
            loss_phoneme += loss_tmp
            count += 1
    loss_phoneme = loss_phoneme / count if count > 0 else 0
    return loss_phoneme

def train():
    args = parse_args()
    config_path = args.config_path
    
    # Create log directory with run name to avoid overriding files
    base_log_dir = None  # Will be set from config
    log_dir = None  # Will be set after loading config
    
    # Load the appropriate config based on resume flag
    if args.resume:
        # When resuming, first check if the run directory exists
        base_log_dir = yaml.safe_load(open(config_path))['training_params']['output_dir']
        log_dir = os.path.join(base_log_dir, args.run_name)
        
        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"Cannot resume training: Run directory '{log_dir}' not found.")
        
        # Use the config from the run directory instead of the provided one
        config_file = os.path.join(log_dir, os.path.basename(config_path))
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Cannot resume training: Config file not found in '{log_dir}'.")
        
        config = yaml.safe_load(open(config_file))
    else:
        # For new runs, use the provided config
        config = yaml.safe_load(open(config_path))
    
    # Extract all necessary parameters from config
    training_params = config['training_params']
    num_steps = training_params['num_steps']
    log_interval = training_params['log_interval']
    save_interval = training_params['save_interval']
    base_log_dir = training_params['output_dir']
    log_dir = os.path.join(base_log_dir, args.run_name)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['preprocess_params']['tokenizer'])
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=training_params['mixed_precision'], split_batches=True, kwargs_handlers=[ddp_kwargs])
    
    # Handle directory setup
    if args.resume:
        accelerator.print(f"Resuming training from '{log_dir}' with existing config.")
    else:
        # Start from scratch - remove existing directory if it exists
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        
        # Create fresh directory
        os.makedirs(log_dir, exist_ok=True)
        # Copy the config file to the run directory
        shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
        accelerator.print(f"Starting new training run in '{log_dir}'.")
    
    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(project="pl-bert", config=config)
    
    # Initialize rolling window queues for losses
    token_losses = deque(maxlen=log_interval)
    phoneme_losses = deque(maxlen=log_interval)
    total_losses = deque(maxlen=log_interval)
    
    # Load the processed dataset from the output directory specified in config
    dataset_path = os.path.join(
        config['preprocess_params']['preprocess_dir'],
        config['preprocess_params']['output_dir']
    )
    dataset = load_from_disk(dataset_path)
    
    batch_size = training_params['batch_size']
    train_dataloader = build_dataloader(
        dataset, validation=False, batch_size=batch_size, num_workers=0, device=accelerator.device, dataset_config=config['dataset_params'])

    albert_base_configuration = AlbertConfig(**config['model_params'])
    
    bert = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(
        bert, num_phonemes=len(symbols), num_tokens=tokenizer.vocab_size, hidden_size=config['model_params']['hidden_size'])
    
    is_checkpoint_found, current_step = find_latest_checkpoint(log_dir)
    
    optimizer = AdamW(bert.parameters(), lr=float(training_params['learning_rate']))
    
    if is_checkpoint_found and args.resume:
        bert, optimizer = load_checkpoint(bert, optimizer, log_dir, current_step, accelerator)
    else:
        current_step = 0
    
    bert, optimizer, train_dataloader = accelerator.prepare(
        bert, optimizer, train_dataloader
    )

    accelerator.print('Start training...')
    for _, batch in enumerate(train_dataloader):        
        current_step += 1
        
        token_ids, phoneme_labels, masked_phonemes, input_lengths, masked_indices = batch
        text_mask = length_to_mask(torch.Tensor(input_lengths)).to(accelerator.device)

        phoneme_pred, token_pred = bert(masked_phonemes, attention_mask=(~text_mask).int())
        
        loss_token = calculate_token_loss(token_pred, token_ids, input_lengths, criterion)
        loss_phoneme = calculate_phoneme_loss(phoneme_pred, phoneme_labels, input_lengths, masked_indices, criterion)

        loss = loss_token + loss_phoneme

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        current_step += 1
        
        # Update rolling window queues
        token_losses.append(loss_token.item())
        phoneme_losses.append(loss_phoneme.item())
        total_losses.append(loss.item())
        
        # Log metrics to wandb
        if accelerator.is_main_process:
            log_dict = {
                "token_loss": loss_token.item(),
                "phoneme_loss": loss_phoneme.item(),
                "total_loss": loss.item(),
            }
            
            # Add rolling window metrics if we have enough data
            if len(total_losses) == log_interval:
                log_dict.update({
                    "token_loss_avg": np.mean(token_losses),
                    "phoneme_loss_avg": np.mean(phoneme_losses),
                    "total_loss_avg": np.mean(total_losses)
                })
                
            wandb.log(log_dict)
            
        if (current_step)%save_interval == 0:
            accelerator.print('Saving..')

            state = {
                'net':  bert.state_dict(),
                'step': current_step,
                'optimizer': optimizer.state_dict(),
            }

            checkpoint_path = os.path.join(log_dir, f"step_{current_step + 1}.pth")
            accelerator.save(state, checkpoint_path)
            accelerator.print(f'Checkpoint saved at: {checkpoint_path}')

        if current_step > num_steps:
            return

if __name__ == "__main__":
    train()