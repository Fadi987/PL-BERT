import os
import torch
import argparse
from transformers import AlbertConfig, AlbertModel, AutoTokenizer
from model import MultiTaskModel
from char_indexer import symbols
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PL-BERT checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.yml file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the HF model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the config
    import yaml
    config = yaml.safe_load(open(args.config_path))
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['preprocess_params']['tokenizer'])
    
    # Initialize the model architecture
    albert_base_configuration = AlbertConfig(**config['model_params'])
    bert = AlbertModel(albert_base_configuration)
    model = MultiTaskModel(
        bert, 
        num_phonemes=len(symbols), 
        num_tokens=tokenizer.vocab_size, 
        hidden_size=config['model_params']['hidden_size']
    )
    
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Remove 'module.' prefix from keys if present (from distributed training)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
    
    # Load the state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the entire model (encoder part)
    model.encoder.save_pretrained(args.output_dir)  # Save the ALBERT part
    
    # Save the full model for tasks that need the task-specific heads
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pl_bert_full_model.pt"))
    
    # Save training metadata
    with open(os.path.join(args.output_dir, "training_metadata.txt"), "w") as f:
        f.write(f"Original checkpoint: {args.checkpoint_path}\n")
        f.write(f"Step: {checkpoint['step']}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
    
    # Save model config
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)
    
    print(f"Model successfully converted and saved to {args.output_dir}")

def load_pl_bert_model(model_dir):
    """
    Load a PL-BERT model that was converted using convert_to_hf.py
    
    Args:
        model_dir: Directory where the converted model was saved
    
    Returns:
        model: The loaded MultiTaskModel
        tokenizer: The associated tokenizer
    """
    # Load the config
    with open(f"{model_dir}/config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['preprocess_params']['tokenizer'])
    
    # Load the encoder (ALBERT) part
    encoder = AlbertModel.from_pretrained(model_dir)
    
    # Initialize the full model
    model = MultiTaskModel(
        encoder,
        num_phonemes=len(symbols),
        num_tokens=tokenizer.vocab_size,
        hidden_size=config['model_params']['hidden_size']
    )
    
    # Load the full model state dict (including task-specific heads)
    full_model_path = f"{model_dir}/pl_bert_full_model.pt"
    model.load_state_dict(torch.load(full_model_path, map_location='cpu'))
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer

if __name__ == "__main__":
    main()