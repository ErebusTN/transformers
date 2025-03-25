"""
Copyright 2025 EGen. All rights reserved.

Licensed under the EGen License, Version 0.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://huggingface.co/ErebusTN/EGen_V1/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tokenization classes for THL-150."""

import json
import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import regex as re
from typing_extensions import Final

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

class TokenizationError(ValueError):
    """Exception raised for tokenization-related errors with contextual information."""
    def __init__(self, message: str, component: str, value: any = None):
        full_msg = f"Tokenization error in {component}: {message}"
        if value is not None:
            full_msg += f" (received {type(value).__name__} {value})"
        super().__init__(full_msg)

VOCAB_FILES_NAMES: Final[Dict[str, str]] = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

MAX_MODEL_INPUT_SIZES: Final[Dict[str, int]] = {"thl/thl-150-tokenizer": 32768}

PRETOKENIZE_REGEX: Final[str] = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

@lru_cache(maxsize=256)
def bytes_to_unicode() -> Dict[int, str]:
    """
    Generates UTF-8 byte to Unicode mapping with validation and optimization.
    
    Returns:
        Dictionary mapping byte values to Unicode characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + 
        list(range(ord("¡"), ord("¬") + 1)) + 
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs.copy()
    
    # Generate mappings for non-printable characters
    bs += [b for b in range(256) if b not in bs]
    cs += [256 + len(cs) - len(bs) + i for i in range(len(bs) - len(cs))]
    
    return {b: chr(c) for b, c in zip(bs, cs)}

def get_pairs(word: str) -> Set[Tuple[str, str]]:
    """
    Generates unique character pairs from a word with validation.
    
    Args:
        word: Input string to process
        
    Returns:
        Set of character pairs
    """
    if not word:
        return set()
    return {(word[i], word[i+1]) for i in range(len(word)-1)}

class THL150Tokenizer(PreTrainedTokenizer):
    """
    Optimized THL-150 tokenizer with enhanced validation and error handling.
    
    Implements byte-level BPE with comprehensive input checks and performance optimizations.
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES

    def __init__(
        self,
        vocab_file: Union[str, Path],
        merges_file: Union[str, Path],
        errors: str = "replace",
        unk_token: Union[str, AddedToken] = "<|endoftext|>",
        bos_token: Optional[Union[str, AddedToken]] = None,
        eos_token: Union[str, AddedToken] = "<|endoftext|>",
        pad_token: Union[str, AddedToken] = "<|endoftext|>",
        clean_up_tokenization_spaces: bool = False,
        split_special_tokens: bool = False,
        **kwargs,
    ):
        """Initialize tokenizer with comprehensive validation."""
        try:
            # Validate file existence
            if not Path(vocab_file).is_file():
                raise TokenizationError("Vocabulary file not found", "init", vocab_file)
            if not Path(merges_file).is_file():
                raise TokenizationError("Merges file not found", "init", merges_file)

            # Validate special tokens
            def validate_token(token: Union[str, AddedToken], name: str) -> AddedToken:
                """Validate and normalize token values."""
                if isinstance(token, str):
                    if not token.strip():
                        raise TokenizationError(f"{name} cannot be empty", "init", token)
                    return AddedToken(token, special=True)
                if not isinstance(token, AddedToken):
                    raise TokenizationError(
                        f"{name} must be str or AddedToken", "init", type(token)
                    )
                return token

            self.bos_token = validate_token(bos_token, "bos_token") if bos_token else None
            self.eos_token = validate_token(eos_token, "eos_token")
            self.unk_token = validate_token(unk_token, "unk_token")
            self.pad_token = validate_token(pad_token, "pad_token")

            # Load vocabulary with validation
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.encoder = self._validate_vocab(json.load(f))
            self.decoder = {v: k for k, v in self.encoder.items()}

            # Load BPE merges with validation
            with open(merges_file, "r", encoding="utf-8") as f:
                bpe_merges = self._load_bpe_merges(f)
            self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

            # Initialize encoding components
            self.errors = errors
            self.byte_encoder = bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            self.pat = re.compile(PRETOKENIZE_REGEX)
            self.cache = {}

            super().__init__(
                errors=errors,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                pad_token=self.pad_token,
                unk_token=self.unk_token,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                split_special_tokens=split_special_tokens,
                **kwargs,
            )

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise TokenizationError("Invalid file format", "init") from e
        except Exception as e:
            logger.critical("Tokenizer initialization failed: %s", str(e))
            raise

    def _validate_vocab(self, vocab: Dict) -> Dict:
        """Validate vocabulary structure and content."""
        if not isinstance(vocab, dict):
            raise TokenizationError("Vocabulary must be a dictionary", "vocab")
        if len(vocab) < 100:
            logger.warning("Unusually small vocabulary (%d tokens)", len(vocab))
        return vocab

    def _load_bpe_merges(self, file_handle) -> List[Tuple[str, str]]:
        """Load and validate BPE merge operations."""
        merges = []
        for line_num, line in enumerate(file_handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if len(tokens := line.split()) != 2:
                raise TokenizationError(
                    f"Invalid merge format at line {line_num}", 
                    "merges",
                    line
                )
            merges.append((tokens[0], tokens[1]))
        if len(merges) < 10:
            raise TokenizationError("Insufficient merge operations", "merges", len(merges))
        return merges

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict:
        return {**self.encoder, **self.added_tokens_encoder}

    @lru_cache(maxsize=10_000)
    def bpe(self, token: str) -> str:
        """Optimized BPE implementation with caching and validation."""
        if not token:
            return token

        word = tuple(token)
        pairs = get_pairs(token)

        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j

                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = get_pairs("".join(word))

        return " ".join(word)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text with input validation and error handling."""
        if not isinstance(text, str):
            raise TokenizationError("Input must be a string", "tokenize", type(text))

        try:
            text = unicodedata.normalize("NFC", text)
            if self.bos_token:
                text = str(self.bos_token) + text

            bpe_tokens = []
            for token in re.findall(self.pat, text):
                encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                bpe_tokens.extend(self.bpe(encoded).split(" "))

            return bpe_tokens
        except UnicodeEncodeError as e:
            raise TokenizationError("Text encoding failed", "tokenize", text[:50]) from e

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(str(self.unk_token)))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string with error recovery."""
        try:
            text = "".join(tokens)
            return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        except (KeyError, UnicodeDecodeError) as e:
            logger.error("Decoding error: %s", str(e))
            return ""  # Graceful degradation

    def save_vocabulary(
        self, 
        save_directory: Union[str, Path],
        filename_prefix: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save tokenizer files with validation and error handling."""
        save_path = Path(save_directory)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        elif not save_path.is_dir():
            raise TokenizationError("Save path must be a directory", "save", save_directory)

        vocab_file = save_path / f"{filename_prefix}-{VOCAB_FILES_NAMES['vocab_file']}"
        merge_file = save_path / f"{filename_prefix}-{VOCAB_FILES_NAMES['merges_file']}"

        try:
            with vocab_file.open("w", encoding="utf-8") as f:
                json.dump(self.encoder, f, indent=2, ensure_ascii=False)

            with merge_file.open("w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
                for pair in sorted(self.bpe_ranks.keys(), key=lambda x: self.bpe_ranks[x]):
                    f.write(f"{pair[0]} {pair[1]}\n")

            return (str(vocab_file), str(merge_file))
        except IOError as e:
            raise TokenizationError("Failed to save vocabulary", "save") from e