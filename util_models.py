import os
import sys
import torch

ROOT_DIR = os.getcwd()
CATT_PATH = os.path.join(ROOT_DIR, "../catt")

sys.path.insert(0, CATT_PATH)
        
from ed_pl import TashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer
from utils import remove_non_arabic

sys.path.remove(CATT_PATH)


class CattTashkeel:
    def __init__(self, device: str = None):
        self.tokenizer = TashkeelTokenizer()
        self.ckpt_path = os.path.join(CATT_PATH, "models/best_ed_mlm_ns_epoch_178.pt")

        print('ckpt_path is:', self.ckpt_path)

        found_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device or found_device
        print("Using device for CattTashkeel: ", self.device)

        self.max_seq_len = 1024
        print('Creating Model...')
        self.model = TashkeelModel(self.tokenizer, max_seq_len=self.max_seq_len, n_layers=3, learnable_pos_emb=False)

        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval().to(self.device)

    def do_tashkeel_batch(self, x, batch_size, verbose):
        x = [remove_non_arabic(i) for i in x]
        x_tashkeel = self.model.do_tashkeel_batch(x, batch_size, verbose)
        return x_tashkeel