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
    
    # Setup configuration and directories
    config, log_dir = setup_config_and_directories(args, config_path)
    
    # Extract training parameters
    training_params = config['training_params']
    num_steps = training_params['num_steps']
    log_interval = training_params['log_interval']
    save_interval = training_params['save_interval']
    max_epochs = 10
    
    # Initialize components
    tokenizer, criterion, accelerator = initialize_components(config, training_params, log_dir, args.resume)
    
    # Initialize wandb and metrics tracking
    token_losses, phoneme_losses, total_losses = initialize_metrics_tracking(accelerator, config, log_interval)
    
    # Setup dataset and dataloader
    train_dataloader, val_dataloader = setup_dataset_and_dataloader(config, accelerator)
    
    # Initialize model
    bert, optimizer, current_step = initialize_model(config, tokenizer, log_dir, args.resume, accelerator)
    
    # Prepare for distributed training
    bert, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        bert, optimizer, train_dataloader, val_dataloader
    )

    # Start training loop
    accelerator.print('Start training...')
    current_step, current_epoch = train_loop(
        bert, optimizer, train_dataloader, val_dataloader, criterion, accelerator,
        current_step, num_steps, save_interval, log_interval,
        token_losses, phoneme_losses, total_losses, log_dir, max_epochs
    )
    
    accelerator.print(f'Training completed at step {current_step}, epoch {current_epoch}')

def setup_config_and_directories(args, config_path):
    """Setup configuration and directories based on resume flag."""
    base_log_dir = None
    log_dir = None
    
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
    
    # Set up log directory
    base_log_dir = config['training_params']['output_dir']
    log_dir = os.path.join(base_log_dir, args.run_name)
    
    # Handle directory setup
    if not args.resume:
        # Start from scratch - remove existing directory if it exists
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        
        # Create fresh directory
        os.makedirs(log_dir, exist_ok=True)
        # Copy the config file to the run directory
        shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    
    return config, log_dir

def initialize_components(config, training_params, log_dir, resume):
    """Initialize tokenizer, loss function, and accelerator."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['preprocess_params']['tokenizer'])
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=training_params['mixed_precision'], 
                             split_batches=True, 
                             kwargs_handlers=[ddp_kwargs])
    
    # Log status message
    if resume:
        accelerator.print(f"Resuming training from '{log_dir}' with existing config.")
    else:
        accelerator.print(f"Starting new training run in '{log_dir}'.")
    
    return tokenizer, criterion, accelerator

def initialize_metrics_tracking(accelerator, config, log_interval):
    """Initialize wandb and metrics tracking queues."""
    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(project="pl-bert", config=config)
    
    # Initialize rolling window queues for losses
    token_losses = deque(maxlen=log_interval)
    phoneme_losses = deque(maxlen=log_interval)
    total_losses = deque(maxlen=log_interval)
    
    return token_losses, phoneme_losses, total_losses

def setup_dataset_and_dataloader(config, accelerator):
    """Load dataset and create dataloader."""
    # Load the processed dataset from the output directory specified in config
    dataset_path = os.path.join(
        config['preprocess_params']['preprocess_dir'],
        config['preprocess_params']['output_dir']
    )
    dataset = load_from_disk(dataset_path)
    
    batch_size = config['training_params']['batch_size']
    train_dataloader, val_dataloader = build_dataloader(
        dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        device=accelerator.device, 
        dataset_config=config['dataset_params']
    )
    
    return train_dataloader, val_dataloader

def initialize_model(config, tokenizer, log_dir, resume, accelerator):
    """Initialize model, optimizer, and load checkpoint if resuming."""
    albert_base_configuration = AlbertConfig(vocab_size=len(symbols), **config['model_params'])
    
    bert = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(
        bert, 
        num_phonemes=len(symbols), 
        num_tokens=tokenizer.vocab_size, 
        hidden_size=config['model_params']['hidden_size']
    )
    
    is_checkpoint_found, current_step = find_latest_checkpoint(log_dir)
    
    optimizer = AdamW(bert.parameters(), lr=float(config['training_params']['learning_rate']))
    
    if is_checkpoint_found and resume:
        bert, optimizer = load_checkpoint(bert, optimizer, log_dir, current_step, accelerator)
    else:
        current_step = 0
    
    return bert, optimizer, current_step

def validate(model, val_dataloader, criterion, accelerator):
    """Run validation and return metrics."""
    model.eval()
    val_token_loss = 0
    val_phoneme_loss = 0
    val_total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            loss_token, loss_phoneme, loss = process_batch(model, batch, criterion, accelerator)
            val_token_loss += loss_token.item()
            val_phoneme_loss += loss_phoneme.item()
            val_total_loss += loss.item()
            num_batches += 1
    
    # Calculate average losses
    val_token_loss /= num_batches
    val_phoneme_loss /= num_batches
    val_total_loss /= num_batches
    
    model.train()
    return val_token_loss, val_phoneme_loss, val_total_loss

def run_validation_and_log(model, val_dataloader, criterion, accelerator, current_step, current_epoch):
    """Run validation and log the results.
    
    Args:
        model: The model to validate
        val_dataloader: Validation dataloader
        criterion: Loss function
        accelerator: Accelerator instance for distributed training
        current_step: Current training step
        current_epoch: Current training epoch
    
    Returns:
        tuple: (val_token_loss, val_phoneme_loss, val_total_loss)
    """
    # Run validation
    val_token_loss, val_phoneme_loss, val_total_loss = validate(
        model, val_dataloader, criterion, accelerator
    )
    
    # Log validation metrics
    if accelerator.is_main_process:
        wandb.log({
            "val_token_loss": val_token_loss,
            "val_phoneme_loss": val_phoneme_loss,
            "val_total_loss": val_total_loss,
            "step": current_step,
            "epoch": current_epoch
        })
        
    accelerator.print(f"Validation at step {current_step}: "
                     f"Token Loss: {val_token_loss:.4f}, "
                     f"Phoneme Loss: {val_phoneme_loss:.4f}, "
                     f"Total Loss: {val_total_loss:.4f}")
    
    return val_token_loss, val_phoneme_loss, val_total_loss

def train_loop(model, optimizer, train_dataloader, val_dataloader, criterion, accelerator, 
               current_step, num_steps, save_interval, log_interval,
               token_losses, phoneme_losses, total_losses, log_dir, max_epochs):
    """Main training loop."""
    current_epoch = 0
    
    while current_epoch < max_epochs:
        current_epoch += 1
        accelerator.print(f'Starting epoch {current_epoch}')
        
        for _, batch in enumerate(train_dataloader):
            # Process batch and compute loss
            loss_token, loss_phoneme, loss = process_batch(model, batch, criterion, accelerator)
            
            # Optimization step
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            current_step += 1
            
            # Update metrics and log
            update_metrics_and_log(
                accelerator, loss_token, loss_phoneme, loss,
                token_losses, phoneme_losses, total_losses,
                log_interval, current_epoch
            )
            
            # Save checkpoint and run validation if needed
            if (current_step) % save_interval == 0:
                save_checkpoint(model, optimizer, current_step, log_dir, accelerator, current_epoch)
                
                # Run validation and log results
                run_validation_and_log(model, val_dataloader, criterion, accelerator, current_step, current_epoch)
            
            # Check if training should end based on steps
            if current_step >= num_steps:
                return current_step, current_epoch
    
    return current_step, current_epoch

def process_batch(model, batch, criterion, accelerator):
    """Process a batch of data and compute losses."""
    token_ids, phoneme_labels, masked_phonemes, input_lengths, masked_indices = batch
    text_mask = length_to_mask(torch.Tensor(input_lengths)).to(accelerator.device)
    
    phoneme_pred, token_pred = model(masked_phonemes, attention_mask=(~text_mask).int())
    
    loss_phoneme = calculate_phoneme_loss(phoneme_pred, phoneme_labels, input_lengths, masked_indices, criterion)
    loss_token = calculate_token_loss(token_pred, token_ids, input_lengths, criterion)
    
    loss = loss_token + loss_phoneme
    
    return loss_token, loss_phoneme, loss

def update_metrics_and_log(accelerator, loss_token, loss_phoneme, loss,
                          token_losses, phoneme_losses, total_losses,
                          log_interval, current_epoch):
    """Update metrics tracking and log to wandb."""
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
            "epoch": current_epoch
        }
        
        # Add rolling window metrics if we have enough data
        if len(total_losses) == log_interval:
            log_dict.update({
                "token_loss_avg": np.mean(token_losses),
                "phoneme_loss_avg": np.mean(phoneme_losses),
                "total_loss_avg": np.mean(total_losses)
            })
            
        wandb.log(log_dict)

def save_checkpoint(model, optimizer, current_step, log_dir, accelerator, current_epoch):
    """Save model checkpoint."""
    accelerator.print('Saving..')
    
    state = {
        'net': model.state_dict(),
        'step': current_step,
        'epoch': current_epoch,
        'optimizer': optimizer.state_dict(),
    }
    
    checkpoint_path = os.path.join(log_dir, f"step_{current_step}.pth")
    accelerator.save(state, checkpoint_path)
    accelerator.print(f'Checkpoint saved at: {checkpoint_path}')

if __name__ == "__main__":
    train()