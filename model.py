import torch
from torch import nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    def __init__(self, model, num_phonemes, num_tokens, hidden_size):
        super().__init__()

        self.encoder = model
        self.phoneme_predictor = nn.Linear(hidden_size, num_phonemes)
        self.token_predictor = nn.Linear(hidden_size, num_tokens)
    
    def forward(self, phonemes, attention_mask=None):
        output = self.encoder(phonemes, attention_mask=attention_mask)
        phoneme_pred = self.phoneme_predictor(output.last_hidden_state)
        token_pred = self.token_predictor(output.last_hidden_state)
        
        return phoneme_pred, token_pred
class PhonemeOnlyModel(nn.Module):
    def __init__(self, model, num_phonemes, hidden_size):
        super().__init__()

        self.encoder = model
        self.phoneme_predictor = nn.Linear(hidden_size, num_phonemes)
    
    def forward(self, phonemes, attention_mask=None):
        output = self.encoder(phonemes, attention_mask=attention_mask)
        phoneme_pred = self.phoneme_predictor(output.last_hidden_state)
        
        return phoneme_pred