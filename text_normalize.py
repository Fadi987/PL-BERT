import numpy as np
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from num2words import num2words
import unicodedata

import re

from char_indexer import PUNCTUATION

from converters.Plain      import Plain
from converters.Punct      import Punct
from converters.Date       import Date
from converters.Letters    import Letters
from converters.Cardinal   import Cardinal
from converters.Verbatim   import Verbatim
from converters.Decimal    import Decimal
from converters.Measure    import Measure
from converters.Money      import Money
from converters.Ordinal    import Ordinal
from converters.Time       import Time
from converters.Electronic import Electronic
from converters.Digit      import Digit
from converters.Fraction   import Fraction
from converters.Telephone  import Telephone
from converters.Address    import Address
from converters.Roman    import Roman
from converters.Range    import Range


months = ['jan',
 'feb',
 'mar',
 'apr',
 'jun',
 'jul',
 'aug',
 'sep',
 'oct',
 'nov',
 'dec',
 'january',
 'february',
 'march',
 'april',
 'june',
 'july',
 'august',
 'september',
 'october',
 'november',
 'december']

labels = {
    "PLAIN": Plain(),
    "PUNCT": Punct(),
    "DATE": Date(),
    "LETTERS": Letters(),
    "CARDINAL": Cardinal(),
    "VERBATIM": Verbatim(),
    "DECIMAL": Decimal(),
    "MEASURE": Measure(),
    "MONEY": Money(),
    "ORDINAL": Ordinal(),
    "TIME": Time(),
    "ELECTRONIC": Electronic(),
    "DIGIT": Digit(),
    "FRACTION": Fraction(),
    "TELEPHONE": Telephone(),
    "ADDRESS": Address(),
    "ROMAN": Roman(),
    "RANGE": Range()
}

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

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

def separate_words_and_punctuation(text):
    """
    Separate text into a list of words and punctuation using regex for better performance.
    Punctuation marks are treated as separate tokens.
    """
    # Create a regex pattern that matches either a punctuation character or a non-space, non-punctuation sequence
    # We escape each punctuation character and join them into a character class
    punct_pattern = '|'.join(re.escape(p) for p in PUNCTUATION)
    pattern = f'({punct_pattern})|([^\s{re.escape("".join(PUNCTUATION))}]+)'
    
    # Find all matches
    tokens = re.findall(pattern, text)
    
    # Flatten the list of tuples and remove empty strings
    result = [t[0] if t[0] else t[1] for t in tokens]
    
    return result

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

word_tokenize = TweetTokenizer().tokenize

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def is_oridinal(inputString):
    return inputString.endswith(("th", "nd", "st", "rd"))

def is_money(inputString):
    return inputString.startswith(('$', '€', '£', '¥'))

def is_time(inputString):
    return ":" in inputString

def is_cardinal(inputString):
    return "," in inputString or len(inputString) <= 3

def is_fraction(inputString):
    return "/" in inputString

def is_decimal(inputString):
    return "." in inputString

def is_range(inputString) : 
    return "-" in inputString

def is_url(inputString):
    return "//" in inputString or ".com" in inputString or ".html" in inputString

def has_month(inputString):
    return inputString.lower() in months or inputString == "May"

def normalize_single(text, prev_text = "", next_text = ""):
    if is_url(text):
        text = labels['ELECTRONIC'].convert(text).upper()
    elif has_numbers(text):
        if has_month(prev_text):
            prev_text = labels['DATE'].get_month(prev_text.lower())
            text = labels['DATE'].convert(prev_text + " " + text).replace(prev_text, "").strip()
        elif has_month(next_text):
            next_text = labels['DATE'].get_month(next_text.lower())
            text = labels['DATE'].convert(text + " " + next_text).replace(next_text, "").strip()
        elif is_oridinal(text):
            text = labels['ORDINAL'].convert(text)
        elif is_time(text):
            text = labels['TIME'].convert(text)
        elif is_money(text):
            text = labels['MONEY'].convert(text)
        elif is_fraction(text):
            text = labels['FRACTION'].convert(text)
        elif is_decimal(text):
            text = labels['DECIMAL'].convert(text)
        elif is_cardinal(text):
            text = labels['CARDINAL'].convert(text)
        elif is_range(text):
            text = labels['RANGE'].convert(text)
        else:
            text = labels['DATE'].convert(text)
        
        if has_numbers(text):
            text = labels['CARDINAL'].convert(text)
    elif text == "#" and has_numbers(next_text):
        text = "number"

    return text.replace("$", "")

def normalize_text(text):
    text = remove_accents(text).replace('–', ' to ').replace('-', ' - ').replace(":p", ": p").replace(":P", ": P").replace(":d", ": d").replace(":D", ": D")
    words = word_tokenize(text)

    df = pd.DataFrame(words, columns=['before'])

    df['after'] = df['before']
    
    df['previous'] = df.before.shift(1).fillna('') + "|" + df.before + "|" + df.before.shift(-1).fillna('')
    
    df['after'] = df['previous'].apply(lambda m: normalize_single(m.split('|')[1], m.split('|')[0], m.split('|')[2]))
    
    return TreebankWordDetokenizer().detokenize(df['after'].tolist()).replace("’ s", "'s").replace(" 's", "'s")

if __name__ == '__main__' : 
    text = 'hello (23 Jan 2020, 12:10 AM)'
    out = normalize_text(text)
    print(out)

