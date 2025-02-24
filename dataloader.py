#coding: utf-8

import random
import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from text_utils import TextCleaner

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from transformers import AutoTokenizer

np.random.seed(1)
random.seed(1)

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset, 
                 tokenizer,
                 word_separator=3039, #TODO: fix this
                 phoneme_separator=" ", 
                 phoneme_mask="M", 
                 max_seq_length=512,
                 word_mask_prob=0.15,
                 phoneme_mask_prob=0.8,
                 replace_prob=0.1):
        
        self.data = dataset
        self.max_seq_length = max_seq_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()
        
        self.word_separator = word_separator
        self.phoneme_separator = phoneme_separator
        self.phoneme_mask = phoneme_mask
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_phonemes = self.data[idx]['phonemes']
        phoneme_str = ''.join(token_phonemes)

        token_ids = self.data[idx]['input_ids']
        output_token_ids = []

        labels = ""
        masked_phonemes = ""

        masked_index = []
        for token_phonemes, token_id in zip(token_phonemes, token_ids):
            output_token_ids.extend([token_id] * len(token_phonemes))
            output_token_ids.append(self.word_separator)
            labels += token_phonemes + " "

            if np.random.rand() < self.word_mask_prob:
                # Choose between no change, masking, or replacement based on probabilities
                choices = ['mask', 'replace', 'no_change']
                probs = [self.phoneme_mask_prob, self.replace_prob, 1-(self.phoneme_mask_prob+self.replace_prob)]
                action = np.random.choice(choices, p=probs)

                if action == 'replace':
                    masked_phonemes += ''.join([phoneme_str[np.random.randint(0, len(phoneme_str))] for _ in range(len(token_phonemes))])  # randomized
                elif action == 'mask':
                    masked_phonemes += self.phoneme_mask * len(token_phonemes) # masked
                else:
                    masked_phonemes += token_phonemes

                if action != 'no_change':
                    masked_index.extend((np.arange(len(masked_phonemes) - len(token_phonemes), len(masked_phonemes))).tolist())
            else:
                masked_phonemes += token_phonemes

            masked_phonemes += self.phoneme_separator

        seq_length = len(masked_phonemes)
        if seq_length > self.max_seq_length:
            random_start = np.random.randint(0, seq_length - self.max_seq_length)
            masked_phonemes = masked_phonemes[random_start:random_start + self.max_seq_length]
            output_token_ids = output_token_ids[random_start:random_start + self.max_seq_length]
            labels = labels[random_start:random_start + self.max_seq_length]
            masked_index = [m-random_start for m in masked_index if m >= random_start and m < random_start + self.max_seq_length]


        masked_phonemes = self.text_cleaner(masked_phonemes) # TODO: fix this
        labels = self.text_cleaner(labels)

        assert len(masked_phonemes) == len(output_token_ids)
        assert len(masked_phonemes) == len(labels)
        
        masked_phonemes = torch.LongTensor(masked_phonemes)
        labels = torch.LongTensor(labels)
        output_token_ids = torch.LongTensor(output_token_ids)
        
        return masked_phonemes, output_token_ids, labels, masked_index
        
class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        max_text_length = max([b[1].shape[0] for b in batch])

        words = torch.zeros((batch_size, max_text_length)).long()
        labels = torch.zeros((batch_size, max_text_length)).long()
        phonemes = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = []
        masked_indices = []
        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            
            text_size = phoneme.size(0)
            words[bid, :text_size] = word
            labels[bid, :text_size] = label
            phonemes[bid, :text_size] = phoneme
            input_lengths.append(text_size)
            masked_indices.append(masked_index)

        return words, labels, phonemes, input_lengths, masked_indices


def build_dataloader(df,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader