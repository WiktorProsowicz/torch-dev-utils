# -*- coding: utf-8 -*-
"""Contains audio processing utilities."""
import dataclasses
import logging
import sys
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeAlias
import itertools
import re

import numpy as np
import gruut
import torch
from transformers import DebertaV2Tokenizer, DebertaV2Model
import nltk


def _logger():
    return logging.getLogger(__name__)


TokenMapping: TypeAlias = List[Tuple[str, List[str]]]


@dataclasses.dataclass
class TextFeatures:
    """Contains features extracted from text."""

    normalized_text: str
    words: List[str]
    word_phoneme_mapping: TokenMapping
    word_bert_mapping: TokenMapping

    def get_phoneme_sequence(self) -> List[str]:
        """Returns the full phoneme sequence for the text."""

        return list(itertools.chain(*[phonemes for _, phonemes in self.word_phoneme_mapping]))

    def get_bert_token_sequence(self) -> List[str]:
        """Returns the full BERT token sequence for the text."""

        return list(itertools.chain(*[tokens for _, tokens in self.word_bert_mapping]))

    def get_word_to_phoneme_spans(self) -> np.ndarray:
        """Returns spans mapping words to phonemes."""

        return np.array([len(phonemes) for _, phonemes in self.word_phoneme_mapping])

    def get_word_to_token_spans(self) -> np.ndarray:
        """Returns spans mapping words to BERT tokens."""

        return np.array([len(tokens) for _, tokens in self.word_bert_mapping])


def add_pauses(text_features: TextFeatures, pauses: List[Tuple[int, str]]):
    """Adds pauses to phonemes and updates word-phoneme spans.

    Args:
        text_features: Text features to modify.
        pauses: List of (word_index, pause_type) tuples.
    """

    for word_idx, pause_type in reversed(pauses):
        text_features.word_phoneme_mapping[word_idx][1].append(pause_type)


def obtain_ling_stats(text_features: TextFeatures) -> torch.Tensor:
    """Calculates phoneme-level linguistic statistics.

    The stats include:
        - Position of the phoneme in the word (normalized to [0, 1])
        - Number of phonemes in the word.
        - Position of the word in the sentence (normalized to [0, 1])
        - Number of words in the sentence.
    """

    ph_pos = []

    for _, phonemes in text_features.word_phoneme_mapping:

        if len(phonemes) == 1:
            ph_pos.append(0.0)
            continue

        for i in range(len(phonemes)):
            ph_pos.append(i / (len(phonemes) - 1))

    ph_num = []

    for _, phonemes in text_features.word_phoneme_mapping:

        ph_num.extend([len(phonemes)] * len(phonemes))

    wd_pos = []

    if len(text_features.word_phoneme_mapping) == 1:
        wd_pos.extend([0.0] * len(text_features.get_phoneme_sequence()))

    else:
        for i, (_, phonemes) in enumerate(text_features.word_phoneme_mapping):
            for _ in phonemes:
                wd_pos.append(i / (len(text_features.word_phoneme_mapping) - 1))

    wd_num = []

    for _, phonemes in text_features.word_phoneme_mapping:

        wd_num.extend([len(text_features.word_phoneme_mapping)] * len(phonemes))

    return torch.tensor([ph_pos, ph_num, wd_pos, wd_num], dtype=torch.float32).T


@dataclasses.dataclass
class _WordStruct:
    """Contains word-level information during text processing.

    Each word represents a punctuation-less unit spanning a certain number of phonemes
    and BERT tokens.
    """

    # Word's textual content.
    text: str
    # List of phonemes in the word.
    phonemes: List[str]
    # Word's text with original punctuation (if any).
    text_with_punct: str


class TextProcessor:
    """Processes text data."""

    # Single punctuation marks to be attached to the leading word.
    single_puncts_replace = {
        ' . ': '. ',
        ' , ': ', ',
        ' ! ': '! ',
        ' ? ': '? ',
        ' " ': '" ',
        ' : ': ': ',
    }

    # Punctuation marks that should be moved outside from the quotation.
    puncts_before_quotes_replace = {
        '."': '".',
        '!"': '"!',
        '?"': '"?',
        ',"': '",',
        ':"': '":',
    }

    # Replacements for post-processing of word structs.
    puncts_after_quotes_replace = {
        '"!': '!"',
        '"?': '?"',
        '."': '."',
    }

    apostrophe_replacements = {
        ",'": ',"',
        ".'": '."',
        "!'": '!"',
        "?'": '?"',
        ":'": ':"',
        " '": ' "',
        "\"'": '""',
        "'\"": '""',
    }

    _GRUUT_PHONEMES = ('oʊ', 'ˌaɪ', 'ˌɑ', 'ˈɑ', 'ˌoʊ', 'p', 'f', 'ˌɛ', 'u', 'eɪ', 'ɡ', 'ɚ',
                       't͡ʃ', 'ˌɔɪ', 'ʃ', 'ˈeɪ', 'ˈɔɪ', 'ð', 'w', 's', 'θ', 'ɪ', 'ˌɪ', 'ŋ', 'ʒ',
                       't', 'ˌaʊ', 'ˌu', 'ɛ', 'ˈɚ', 'ˌɔ', 'ɔɪ', 'ɑ', 'ˈi', 'h', 'ɹ', 'ˈɪ', 'j',
                       'ˌʊ', 'm', 'ɔ', 'ˈoʊ', 'æ', 'z', 'i', 'ˌeɪ', 'ˈʊ', 'ˌʌ', 'ˌɚ', 'ˈaʊ', 'b',
                       'd', 'v', 'd͡ʒ', 'ʌ', 'ˈu', 'ˌi', 'l', 'aɪ', 'aʊ', 'ə', 'ˈɔ',
                       'ˈɛ', 'n', 'ˈæ', 'ˌæ', 'ˈaɪ', 'k', 'ʊ', 'ˈʌ')

    _PHONEME_PAUSE_TOKENS = ('<short_pause>', '<medium_pause>', '<long_pause>')

    _NLTK_POS_TAGS = ('LS', 'TO', 'VBN', 'WP', 'UH', 'VBG', 'JJ', 'VBZ', 'VBP', 'NN', 'DT', 'PRP',
                      'WP$', 'NNPS', 'PRP$', 'WDT', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP',
                      'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC',
                      'CD', 'POS')

    SUPPORTED_PHONEMES = _PHONEME_PAUSE_TOKENS + _GRUUT_PHONEMES

    allowed_chars = (
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'abcdefghijklmnopqrstuvwxyz'
        '0123456789'
        " .,:!?'\"$€"
    )

    allowed_chars_for_word_repr = (
        'abcdefghijklmnopqrstuvwxyz'
        "'"
    )

    def __init__(self, tokenizer_tag: str):
        """Inits the text processor."""

        self._tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_tag)

        self._phoneme_to_id = {p: i for i, p in enumerate(self.SUPPORTED_PHONEMES, start=1)}
        self._pos_to_id = {t: i for i, t in enumerate(self._NLTK_POS_TAGS, start=1)}

        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans the text by removing unwanted characters and fixing quotes."""

        if text.startswith("'"):
            text = '"' + text[1:]

        if text.endswith("'"):
            text = text[:-1] + '"'

        for pattern, replacement in TextProcessor.apostrophe_replacements.items():
            text = text.replace(pattern, replacement)

        text = text.replace('-', ' ')

        text = ''.join(filter(lambda x: x in TextProcessor.allowed_chars, text))
        text = ' '.join(text.split())

        return text

    @staticmethod
    def load_text(text_path: str) -> str:
        """Loads text from a file."""

        with open(text_path, 'r', encoding='utf-8') as text_f:
            return text_f.read().strip()

    @staticmethod
    def get_pause_type(pause_length: float) -> str:
        """Returns the type of pause based on its length."""

        if pause_length < 0.2:
            return '<short_pause>'

        if pause_length < 0.7:
            return '<medium_pause>'

        return '<long_pause>'

    def tokenize_text(self, text: str) -> TextFeatures:
        """Processes and tokenizes text."""

        normalized_text = self._prepare_for_tokenization(text)

        word_structs = self._get_word_structs(normalized_text)

        words = []
        word_phoneme_mapping: TokenMapping = []
        word_bert_mapping: TokenMapping = []

        for word_struct in word_structs:
            words.append(word_struct.text)
            word_phoneme_mapping.append((word_struct.text, word_struct.phonemes))

            tokens = self._tokenizer.tokenize(word_struct.text_with_punct)
            word_bert_mapping.append((word_struct.text, tokens))

        return TextFeatures(
            normalized_text=normalized_text,
            words=words,
            word_phoneme_mapping=word_phoneme_mapping,
            word_bert_mapping=word_bert_mapping)

    def obtain_pos_tags(self, text_features: TextFeatures) -> List[int]:
        """Returns a sequence of POS tag IDs for the words in the text."""

        pos_tags_with_words = nltk.pos_tag(text_features.words, lang='eng')
        pos_tags = [tag for _, tag in pos_tags_with_words]

        try:
            return [self._pos_to_id[tag] for tag in pos_tags]

        except KeyError:
            _logger().warning('Unsupported POS tag found in tags: %s', pos_tags_with_words)

        return [self._pos_to_id.get(tag, 0) for tag in pos_tags]

    def obtain_bert_tokens_for_sentence(self, sentence: str) -> List[str]:
        """Obtains BERT tokens for a given sentence."""

        sentence = self._prepare_for_tokenization(sentence)
        return self._tokenizer.tokenize(sentence)

    def obtain_phoneme_ids(self, phonemes: List[str]) -> List[int]:
        """Converts phonemes to their corresponding IDs."""

        try:
            return [self._phoneme_to_id[p] for p in phonemes]

        except KeyError:
            _logger().critical('Unsupported phoneme found in the text: %s', phonemes)
            sys.exit(1)

    def _prepare_for_tokenization(self, text: str) -> str:
        """Cleans the text and prepares it for tokenization."""

        text = self.clean_text(text)

        def scream_case_replacer(match: re.Match) -> str:
            return match.group(0).capitalize()

        text = re.sub(r'\b[A-Z]{2,}\b', scream_case_replacer, text)

        for pattern, replacement in self.single_puncts_replace.items():
            text = text.replace(pattern, replacement)

        return text

    def _get_word_structs(self, text: str) -> List[_WordStruct]:
        """Converts text to a list of word structs."""

        for pattern, replacement in self.puncts_before_quotes_replace.items():
            text = text.replace(pattern, replacement)

        text = text.replace('"', '`')

        word_structs: List[_WordStruct] = []

        for sentence in gruut.sentences(text,
                                        lang='en-us',
                                        punctuations=True,
                                        phonemes=True):
            for word in sentence.words:
                if not word.phonemes:
                    continue

                if word.is_break and not word_structs:
                    continue

                if word.is_break:
                    word_structs[-1].text_with_punct += word.text
                    continue

                word_structs.append(_WordStruct(
                    text=self._get_proper_word_representation(word.text.lower()),
                    phonemes=list(word.phonemes),
                    text_with_punct=word.text.replace('`', '"')))

        self._post_process_word_structs(word_structs)

        return word_structs

    def _get_proper_word_representation(self, word: str) -> str:
        """Converts a word to its proper representation for tokenization."""

        return ''.join(filter(lambda x: x in TextProcessor.allowed_chars_for_word_repr, word))

    def _post_process_word_structs(self, word_structs: List[_WordStruct]):
        """Post-processes word structs to fix punctuation placement."""

        for word_struct in word_structs:

            text_with_punct = word_struct.text_with_punct

            for pattern, replacement in self.puncts_after_quotes_replace.items():
                text_with_punct = text_with_punct.replace(pattern, replacement)

            word_struct.text_with_punct = text_with_punct
