"""
Text conversion and vocabulary mapping for MambaFlow-TTS.
"""

from preprocessing.text import cleaners
from preprocessing.text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


class UnknownCleanerException(Exception):
    pass


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through (e.g. ['english_cleaners2'])
    Returns:
      sequence: List of integers corresponding to the symbols in the text
      clean_text: Cleaned/phonemized string
    """
    sequence = []
    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        symbol_id = _symbol_to_id.get(symbol, None)
        if symbol_id is not None:
            sequence.append(symbol_id)
    return sequence, clean_text


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of already cleaned/phonemized text to symbol IDs."""
    sequence = []
    for symbol in cleaned_text:
        symbol_id = _symbol_to_id.get(symbol, None)
        if symbol_id is not None:
            sequence.append(symbol_id)
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string."""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol.get(symbol_id, "")
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if not cleaner:
            raise UnknownCleanerException(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text
