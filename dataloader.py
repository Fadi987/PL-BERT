#coding: utf-8

import random
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Subset

from char_indexer import CharacterIndexer, PHONEME_MASK, PHONEME_SEPARATOR, PUNCTUATION

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

class MaskedPhonemeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, word_pred_prob, phoneme_mask_prob, 
                 replace_prob, word_separator, max_seq_length, use_token_ids):

        self.data = dataset
        self.max_seq_length = max_seq_length
        self.word_pred_prob = word_pred_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.char_indexer = CharacterIndexer()
        self.word_separator = word_separator
        self.use_token_ids = use_token_ids
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_phonemes = self.data[idx]['phonemes']
        phoneme_str = ''.join(token_phonemes)
        
        # Get token_ids if available and needed, otherwise use placeholder
        token_ids = self.data[idx]['token_ids'] if self.use_token_ids else [self.word_separator] * len(token_phonemes)
        
        # Process tokens to create masked phonemes and labels
        output_token_ids, phoneme_labels, masked_phonemes, masked_index = self._process_tokens(
            token_phonemes, token_ids, phoneme_str)
        
        # Handle sequence length and truncation if needed
        masked_phonemes, output_token_ids, phoneme_labels, masked_index = self._handle_sequence_length(
            masked_phonemes, output_token_ids, phoneme_labels, masked_index)
        
        # Convert to tensor format
        masked_phonemes, phoneme_labels, output_token_ids = self._convert_to_tensors(
            masked_phonemes, phoneme_labels, output_token_ids)
        
        if self.use_token_ids:
            return output_token_ids, phoneme_labels, masked_phonemes, masked_index
        else:
            return phoneme_labels, masked_phonemes, masked_index
    
    def _process_tokens(self, token_phonemes, token_ids, phoneme_str):
        """Process tokens to create masked phonemes, labels, and track masked indices."""
        output_token_ids = []
        phoneme_labels = ""
        masked_phonemes = ""
        masked_index = []
        
        for token_phonemes, token_id in zip(token_phonemes, token_ids):
            output_token_ids.extend([token_id] * len(token_phonemes))
            output_token_ids.append(self.word_separator)
            phoneme_labels += token_phonemes + PHONEME_SEPARATOR
            
            # Apply masking strategy
            token_masked_phonemes, token_masked_index = self._apply_masking_strategy(
                token_phonemes, phoneme_str, len(masked_phonemes))
            
            masked_phonemes += token_masked_phonemes
            masked_phonemes += PHONEME_SEPARATOR
            
            if token_masked_index:
                masked_index.extend(token_masked_index)
                
        return output_token_ids, phoneme_labels, masked_phonemes, masked_index
    
    def _apply_masking_strategy(self, token_phonemes, phoneme_str, current_length):
        """Apply masking strategy to a token's phonemes."""
        if np.random.rand() < self.word_pred_prob:
            # Choose between no change, masking, or replacement based on probabilities
            choices = ['mask', 'replace', 'no_change']
            probs = [self.phoneme_mask_prob, self.replace_prob, 1-(self.phoneme_mask_prob+self.replace_prob)]
            action = np.random.choice(choices, p=probs)
            
            if action == 'replace':
                # WARNING: we're doing random replacement of phonemes of the current text which is not ideal. 
                # Ideally, we should randomize over the entire phoneme vocabulary
                masked_phonemes = ''.join(random.choices(phoneme_str, k=len(token_phonemes)))
            elif action == 'mask':
                masked_phonemes = PHONEME_MASK * len(token_phonemes)  # masked
            else:  # no_change
                masked_phonemes = token_phonemes
                
            # Track masked indices if we modified the phonemes
            if action != 'no_change':
                masked_index = (np.arange(current_length + len(masked_phonemes) - len(token_phonemes), 
                                         current_length + len(masked_phonemes))).tolist()
                return masked_phonemes, masked_index
            
            return masked_phonemes, []
        else:
            return token_phonemes, []
    
    def _handle_sequence_length(self, masked_phonemes, output_token_ids, phoneme_labels, masked_index):
        """Handle sequence length and truncation if needed."""
        seq_length = len(masked_phonemes)
        if seq_length > self.max_seq_length:
            random_start = np.random.randint(0, seq_length - self.max_seq_length)
            end = random_start + self.max_seq_length
            
            # Slice all sequences at once
            masked_phonemes = masked_phonemes[random_start:end]
            output_token_ids = output_token_ids[random_start:end]
            phoneme_labels = phoneme_labels[random_start:end]
            
            # Filter and adjust masked indices in one step
            masked_index = [idx - random_start for idx in masked_index 
                           if random_start <= idx < end]
        
        return masked_phonemes, output_token_ids, phoneme_labels, masked_index
    
    def _convert_to_tensors(self, masked_phonemes, phoneme_labels, output_token_ids):
        """Convert processed data to PyTorch tensors."""
        # Convert characters to indices
        masked_phonemes = self.char_indexer(masked_phonemes)
        phoneme_labels = self.char_indexer(phoneme_labels)
        
        # Convert to tensors
        masked_phonemes = torch.LongTensor(masked_phonemes)
        phoneme_labels = torch.LongTensor(phoneme_labels)
        output_token_ids = torch.LongTensor(output_token_ids)

        assert len(masked_phonemes) == len(output_token_ids)
        assert len(masked_phonemes) == len(phoneme_labels)
        
        return masked_phonemes, phoneme_labels, output_token_ids

class TruncatedTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_seq_length):
        self.data = dataset
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Handle sequence length and truncation if needed
        truncated_text = self._truncate_text_if_needed(text)
        
        return {
            'id': self.data[idx]['id'],
            'url': self.data[idx]['url'],
            'title': self.data[idx]['title'],
            'text': truncated_text
        }
    
    def _truncate_text_if_needed(self, text):
        """Truncate text to max_seq_length ensuring complete sentences are preserved."""
        seq_length = len(text)
        if seq_length <= self.max_seq_length:
            return text
            
        
        # Find a random starting point
        random_start = np.random.randint(0, max(1, seq_length - self.max_seq_length))
        
        # Adjust start to beginning of a sentence if possible
        if random_start > 0:
            # Look backward for the nearest sentence boundary
            for i in range(random_start - 1, -1, -1):
                if text[i] in PUNCTUATION:
                    random_start = i + 1  # Start after the punctuation
                    break
        
        # Calculate potential end position
        end = min(random_start + self.max_seq_length, seq_length)
        
        # Adjust end to complete the last sentence if possible
        if end < seq_length:
            # Look forward for the nearest sentence boundary
            for i in range(end, min(seq_length, end + int(0.2 * self.max_seq_length))):
                if text[i] in PUNCTUATION:
                    end = i + 1  # Include the punctuation
                    break
                    
        # Sample a sequence of sentences
        text = text[random_start:end]
        
        return text
    

class Collater(object):
    def __call__(self, batch):
        batch_size = len(batch)

        # sort by sequence length (descending order)
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        max_text_length = batch[0][0].shape[0]

        batch_token_ids = torch.zeros((batch_size, max_text_length)).long()
        batch_phoneme_labels = torch.zeros((batch_size, max_text_length)).long()
        batch_masked_phonemes = torch.zeros((batch_size, max_text_length)).long()

        input_lengths = [0] * batch_size
        batch_masked_indices = [None] * batch_size

        for idx, (token_ids, phoneme_labels, masked_phonemes, masked_indices) in enumerate(batch):
            text_size = masked_phonemes.size(0)
            batch_token_ids[idx, :text_size] = token_ids
            batch_phoneme_labels[idx, :text_size] = phoneme_labels
            batch_masked_phonemes[idx, :text_size] = masked_phonemes
            input_lengths[idx] = text_size
            batch_masked_indices[idx] = masked_indices

        return batch_token_ids, batch_phoneme_labels, batch_masked_phonemes, input_lengths, batch_masked_indices

def build_dataloader(df, batch_size, device, dataset_config, use_token_ids, num_workers=0, **kwargs):
    # Create the full dataset
    dataset = MaskedPhonemeDataset(df, use_token_ids=use_token_ids, **dataset_config)
    
    # Calculate validation size (min of 5% of dataset and 10000)
    total_size = len(dataset)
    val_size = min(int(total_size * 0.05), 10000)
    train_size = total_size - val_size
    
    # Create random indices for train and validation splits
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Use appropriate collator based on whether we're using token_ids
    if use_token_ids:
        collate_fn = Collater()
    else:
        collate_fn = PhonemeOnlyCollater()
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=collate_fn, 
        pin_memory=(device != 'cpu'),
        num_workers=num_workers,
        **kwargs
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        collate_fn=collate_fn, 
        pin_memory=(device != 'cpu'),
        num_workers=num_workers,
        **kwargs
    )
    
    return train_loader, val_loader

class PhonemeOnlyCollater(object):
    def __call__(self, batch):
        batch_size = len(batch)

        # sort by sequence length (descending order)
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        max_text_length = batch[0][0].shape[0]

        batch_phoneme_labels = torch.zeros((batch_size, max_text_length)).long()
        batch_masked_phonemes = torch.zeros((batch_size, max_text_length)).long()

        input_lengths = [0] * batch_size
        batch_masked_indices = [None] * batch_size

        for idx, (phoneme_labels, masked_phonemes, masked_indices) in enumerate(batch):
            text_size = masked_phonemes.size(0)
            batch_phoneme_labels[idx, :text_size] = phoneme_labels
            batch_masked_phonemes[idx, :text_size] = masked_phonemes
            input_lengths[idx] = text_size
            batch_masked_indices[idx] = masked_indices

        return batch_phoneme_labels, batch_masked_phonemes, input_lengths, batch_masked_indices