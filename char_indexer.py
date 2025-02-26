# IPA Phonemizer: https://github.com/bootphon/phonemizer

PAD = "$"
PUNCTUATION = ';:,.!?¡¿—…"«»“”'
LETTERS = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئى'
LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩ᵻ"
# NOTE: 'U' is a valid 'unknown' character because it is different from all the characters above it. In English PL-BERT that was not the case which was not ideal
PHONEME_MASK = "M"
PHONEME_SEPARATOR = " "
UNKNOWN='U'

# Export all symbols:
symbols = [PAD] + list(PUNCTUATION) + list(LETTERS) + list(LETTERS_IPA) + [PHONEME_MASK] + [PHONEME_SEPARATOR] + [UNKNOWN]

assert len(symbols) == len(set(symbols)) # no duplicates

class CharacterIndexer:
    def __init__(self):
        self.word_index_dictionary = {symbol: i for i, symbol in enumerate(symbols)}

    def __call__(self, text):
        return [self.word_index_dictionary[char] if char in self.word_index_dictionary 
                else self.word_index_dictionary[UNKNOWN] for char in text]