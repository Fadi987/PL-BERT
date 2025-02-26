# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئى'
phoneme_str = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩ᵻ"
# NOTE: 'U' is a valid 'unknown' character because it is different from all the characters above it. In English PL-BERT that was not the case which was not ideal
_unknown='U'

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [_unknown]

assert len(symbols) == len(set(symbols)) # no duplicates

class CharacterIndexer:
    def __init__(self):
        self.word_index_dictionary = {symbol: i for i, symbol in enumerate(symbols)}

    def __call__(self, text):
        return [self.word_index_dictionary[char] if char in self.word_index_dictionary 
                else self.word_index_dictionary[_unknown] for char in text]