# Standard library imports
import os
import shutil
import yaml

# Third-party imports
import torch
from torch import nn
from datasets import load_from_disk

# Accelerate imports
from accelerate import Accelerator, DistributedDataParallelKwargs

# Transformers imports
from transformers import AdamW, AlbertConfig, AlbertModel, AutoTokenizer

# Local imports
from dataloader import build_dataloader
from model import MultiTaskModel
from char_indexer import symbols

config_path = "external/pl_bert/configs/config.yml" # you can change it to anything else
config = yaml.safe_load(open(config_path))
training_params = config['training_params']

tokenizer = AutoTokenizer.from_pretrained(config['preprocess_params']['tokenizer'])

criterion = nn.CrossEntropyLoss() # F0 loss (regression)

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
loss_train_record = list([])
loss_test_record = list([])

num_steps = training_params['num_steps']
log_interval = training_params['log_interval']
save_interval = training_params['save_interval']

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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=training_params['mixed_precision'], split_batches=True, kwargs_handlers=[ddp_kwargs])
    
    curr_steps = 0
    
    # Load the processed dataset from the output directory specified in config
    dataset_path = os.path.join(
        config['preprocess_params']['preprocess_dir'],
        config['preprocess_params']['output_dir']
    )
    dataset = load_from_disk(dataset_path)

    log_dir = training_params['output_dir']
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    
    batch_size = training_params['batch_size']
    train_dataloader = build_dataloader(
        dataset, validation=False, batch_size=batch_size, num_workers=0, device=accelerator.device, dataset_config=config['dataset_params'])

    albert_base_configuration = AlbertConfig(**config['model_params'])
    
    bert = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(
        bert, num_phonemes=len(symbols), num_tokens=tokenizer.vocab_size, hidden_size=config['model_params']['hidden_size'])
    
    is_checkpoint_found, current_step = find_latest_checkpoint(log_dir)
    
    optimizer = AdamW(bert.parameters(), lr=float(training_params['learning_rate']))
    
    if is_checkpoint_found:
        bert, optimizer = load_checkpoint(bert, optimizer, log_dir, current_step, accelerator)
    
    bert, optimizer, train_dataloader = accelerator.prepare(
        bert, optimizer, train_dataloader
    )

    accelerator.print('Start training...')

    running_loss = 0
    for _, batch in enumerate(train_dataloader):        
        curr_steps += 1
        
        token_ids, phoneme_labels, masked_phonemes, input_lengths, masked_indices = batch
        text_mask = length_to_mask(torch.Tensor(input_lengths)).to(accelerator.device)

        phoneme_pred, token_pred = bert(masked_phonemes, attention_mask=(~text_mask).int())
        
        loss_token = calculate_token_loss(token_pred, token_ids, input_lengths, criterion)
        loss_phoneme = calculate_phoneme_loss(phoneme_pred, phoneme_labels, input_lengths, masked_indices, criterion)

        loss = loss_token + loss_phoneme

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        running_loss += loss.item()

        current_step += 1
        if (current_step)%log_interval == 0:
            accelerator.print ('Step [%d/%d], Loss: %.5f, Token Loss: %.5f, Phoneme Loss: %.5f'
                    %(current_step, num_steps, running_loss / log_interval, loss_token, loss_phoneme))
            running_loss = 0
            
        if (current_step)%save_interval == 0:
            accelerator.print('Saving..')

            state = {
                'net':  bert.state_dict(),
                'step': current_step,
                'optimizer': optimizer.state_dict(),
            }

            accelerator.save(state, log_dir + '/step_' + str(current_step + 1) + '.pth')

        if curr_steps > num_steps:
            return

if __name__ == "__main__":
    train()