import string
from text_normalize import normalize_text, remove_accents

def phonemize(text, global_phonemizer, tokenizer):
    text = normalize_text(remove_accents(text))
    tokens = tokenizer.tokenize(text)

    tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    phonemes = [global_phonemizer.phonemize([token.replace("#", "")], strip=True)[0] if token not in string.punctuation else token for token in tokens]
        
    return {'tokens_ids' : tokens_ids, 'phonemes': phonemes}
