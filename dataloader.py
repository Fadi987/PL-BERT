#coding: utf-8

import random
import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from char_indexer import CharacterIndexer, PHONEME_MASK, PHONEME_SEPARATOR

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

class MaskedPhonemeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, word_pred_prob, phoneme_mask_prob, 
                 replace_prob, word_separator, max_seq_length):

        self.data = dataset
        self.max_seq_length = max_seq_length
        self.word_pred_prob = word_pred_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.char_indexer = CharacterIndexer()
        
        self.word_separator = word_separator
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_phonemes = self.data[idx]['phonemes']
        phoneme_str = ''.join(token_phonemes)
        token_ids = self.data[idx]['token_ids']
        
        # Process tokens to create masked phonemes and labels
        output_token_ids, phoneme_labels, masked_phonemes, masked_index = self._process_tokens(
            token_phonemes, token_ids, phoneme_str)
        
        # Handle sequence length and truncation if needed
        masked_phonemes, output_token_ids, phoneme_labels, masked_index = self._handle_sequence_length(
            masked_phonemes, output_token_ids, phoneme_labels, masked_index)
        
        # Convert to tensor format
        masked_phonemes, phoneme_labels, output_token_ids = self._convert_to_tensors(
            masked_phonemes, phoneme_labels, output_token_ids)
        
        return masked_phonemes, output_token_ids, phoneme_labels, masked_index
    
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

        for idx, (masked_phonemes, token_ids, phoneme_labels, masked_indices) in enumerate(batch):
            text_size = masked_phonemes.size(0)
            batch_token_ids[idx, :text_size] = token_ids
            batch_phoneme_labels[idx, :text_size] = phoneme_labels
            batch_masked_phonemes[idx, :text_size] = masked_phonemes
            input_lengths[idx] = text_size
            batch_masked_indices[idx] = masked_indices

        return batch_token_ids, batch_phoneme_labels, batch_masked_phonemes, input_lengths, batch_masked_indices

def build_dataloader(df, validation, device, dataset_config, **kwargs):
    dataset = MaskedPhonemeDataset(df, **dataset_config)
    collate_fn = Collater()
    data_loader = DataLoader(dataset, shuffle=(not validation), drop_last=(not validation), collate_fn=collate_fn, pin_memory=(device != 'cpu'), **kwargs)

    return data_loader