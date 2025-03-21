import os
from collections import Counter
from datasets import load_from_disk
import json
from tqdm.auto import tqdm

# TODO: think about usage fo [PAD], [CLS], [SEP], [MASK] tokens
# TODO: think about size of the vocab (current 60000)
# TODO: think about how much coverage we have (i.e, what percentage of the dataset is not UNK)
# TODO: plot frequency histogram of the words
# TODO: think about including punctuation into the vocab

def create_custom_tokenizer(dataset_path, vocab_size=60000, output_dir="custom_tokenizer"):
    """
    Create a custom word tokenizer for the Arabic Wikipedia dataset.
    
    Args:
        dataset_path: Path to the HuggingFace dataset
        vocab_size: Size of vocabulary to keep (most frequent words)
        output_dir: Directory to save the tokenizer files
    
    Returns:
        Dictionary with vocabulary and token mappings
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Count word frequencies
    print("Counting word frequencies...")
    word_counter = Counter()
    
    for example in tqdm(dataset, desc="Processing texts"):
        # Simple Arabic word tokenization (split on whitespace)
        # You might want to use a more sophisticated approach for Arabic
        words = example['text'].split()
        word_counter.update(words)
    
    print(f"Total unique words found: {len(word_counter)}")
    
    # Create vocabulary with most common words
    vocab = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]  # Special tokens
    vocab.extend([word for word, _ in word_counter.most_common(vocab_size - len(vocab))])
    
    # Create token mappings
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    
    # Create tokenizer dictionary
    tokenizer_dict = {
        "vocab": vocab,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(vocab)
    }
    
    # Save tokenizer files
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Tokenizer saved to {output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return tokenizer_dict

# Tokenizer class for using the custom tokenizer
class ArabicWordTokenizer:
    def __init__(self, tokenizer_path):
        """
        Initialize the tokenizer from saved files.
        
        Args:
            tokenizer_path: Path to the tokenizer directory
        """
        with open(os.path.join(tokenizer_path, "tokenizer.json"), "r", encoding="utf-8") as f:
            tokenizer_dict = json.load(f)
        
        self.vocab = tokenizer_dict["vocab"]
        self.token_to_id = tokenizer_dict["token_to_id"]
        self.id_to_token = {int(k): v for k, v in tokenizer_dict["id_to_token"].items()}
        self.vocab_size = tokenizer_dict["vocab_size"]
        self.unk_token_id = self.token_to_id["[UNK]"]
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        # For Arabic, you might want to use a more sophisticated approach
        return text.split()
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to token IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """
        Convert token IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        return [self.id_to_token.get(id, "[UNK]") for id in ids]
    
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    
    def decode(self, ids):
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = self.convert_ids_to_tokens(ids)
        return " ".join(tokens)

# Example usage
if __name__ == "__main__":
    # Create the tokenizer
    tokenizer_dict = create_custom_tokenizer(
        dataset_path="path/to/wikipedia_dataset",
        vocab_size=60000,
        output_dir="arabic_wiki_tokenizer"
    )
    
    # Load and use the tokenizer
    tokenizer = ArabicWordTokenizer("arabic_wiki_tokenizer")
    
    # Test the tokenizer
    sample_text = "مرحبا بالعالم"
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)
    
    print(f"Sample text: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")