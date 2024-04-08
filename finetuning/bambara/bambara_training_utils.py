import os
import re
from typing import List, Union, Dict

import torch
from tokenizers import Tokenizer
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer
from coqpit import Coqpit
from tqdm import tqdm


def bambara_dataset_formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in tqdm(ttf):
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            speaker_name = f"speaker_{cols[3]}".replace("\n", "")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


class VoiceBambaraTextPreprocessor:
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(text) for text in texts]

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = self.expand_number(text)

        return text

    def expand_number(self, text):
        """
        Normalize Bambara text for TTS by replacing numerical figures with their word equivalents.

        Args:
        text (str): The text to be normalized.

        Returns:
        str: The normalized Bambara text.
        """

        # A regex pattern to match all numbers
        number_pattern = re.compile(r'\b\d+\b')

        # Function to replace each number with its Bambara text
        def replace_number_with_text(match):
            number = int(match.group())
            return self.number_to_bambara(number)

        # Replace each number in the text with its Bambara word equivalent
        normalized_text = number_pattern.sub(replace_number_with_text, text)

        return normalized_text

    def number_to_bambara(self, n):

        """
        Convert a number into its textual representation in Bambara using recursion.
        Args:
        n (int): The number to be converted.
        Returns:
        str: The number expressed in Bambara text.
        Examples:
        >>> number_to_bambara(123)
        'kɛmɛ ni mugan ni saba'
        Notes:
        This function assumes that 'n' is a non-negative integer.
        """

        # Bambara numbering rules
        units = ["", "kɛlɛn", "fila", "saba", "naani", "duuru", "wɔrɔ", "wòlonwula", "sɛɛgin", "kɔnɔntɔn"]
        tens = ["", "tan", "mugan", "bisaba", "binaani", "biduuru", "biwɔrɔ", "biwòlonfila", "bisɛɛgin", "bikɔnɔntɔn"]
        hundreds = ["", "kɛmɛ"]
        thousands = ["", "waga"]
        millions = ["", "milyɔn"]

        # Handle zero explicitly
        if n == 0:
            return ""  # bambara does not support zero

        if n < 10:
            return units[n]
        elif n < 100:
            return tens[n // 10] + (" ni " + self.number_to_bambara(n % 10) if n % 10 > 0 else "")
        elif n < 1000:
            return hundreds[1] + (" " + self.number_to_bambara(n // 100) if n >= 200 else "") + (
                " ni " + self.number_to_bambara(n % 100) if n % 100 > 0 else "")
        elif n < 1_000_000:
            return thousands[1] + " " + self.number_to_bambara(n // 1000) + (
                " ni " + self.number_to_bambara(n % 1000) if n % 1000 > 0 else "")
        else:
            return millions[1] + " " + self.number_to_bambara(n // 1_000_000) + (
                " ni " + self.number_to_bambara(n % 1_000_000) if n % 1_000_000 > 0 else "")


class VoiceBambaraTokenizer:
    def __init__(self, vocab_file=None):
        self.tokenizer = None
        self.text_preprocessor = VoiceBambaraTextPreprocessor()
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt, lang):
        return self.text_preprocessor.preprocess(txt)

    def encode(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        txt = self.preprocess_text(txt, lang)
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")

        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1


class BambaraGPTTrainer(GPTTrainer):
    def __init__(self, config: Coqpit):
        super().__init__(config)

        # create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBambaraTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.xtts.init_models()
        
    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return BambaraGPTTrainer(config)
