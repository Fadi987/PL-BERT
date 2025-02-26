import re
import string
from text_normalize import normalize_text, remove_accents

from num2words import num2words

def convert_numbers_to_arabic_words(text):
    """Convert English numerals in Arabic text to Arabic word form."""
    # Find all numbers in the text with word boundaries
    numbers = re.findall(r'\d+', text)
    
    # Sort numbers by length in descending order to avoid partial replacements
    # (e.g., replacing "19" in "1986" before replacing "1986" itself)
    numbers.sort(key=len, reverse=True)
    
    # Replace each number with its Arabic word form
    for num in numbers:
        try:
            # Convert to integer
            n = int(num)
            # Use num2words with Arabic language
            arabic_word = num2words(n, lang='ar')
            # Replace the number with its word form using word boundaries
            text = re.sub(re.escape(num), arabic_word, text)
        except (ValueError, NotImplementedError):
            # Skip if conversion fails
            continue
    
    return text

def filter_non_arabic_words(text):
    """Remove non-Arabic words from text."""
    # Arabic Unicode range (includes Arabic, Persian, Urdu characters)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0660-\u0669]+')
    
    # Split text into words
    words = text.split()
    
    # Keep only words that contain Arabic characters
    arabic_words = []
    for word in words:
        # Check if the word ONLY contains Arabic characters
        if arabic_pattern.search(word):
            arabic_words.append(word)
    
    # Join the Arabic words back into text
    return ' '.join(arabic_words)

def phonemize(text, global_phonemizer, tokenizer):
    text = convert_numbers_to_arabic_words(text)
    text = filter_non_arabic_words(text)

    text = normalize_text(remove_accents(text))
    tokens = tokenizer.tokenize(text)

    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    phonemes = [global_phonemizer.phonemize([token.replace("#", "")], strip=True)[0] if token not in string.punctuation else token for token in tokens]
        
    return {'token_ids' : token_ids, 'phonemes': phonemes}
