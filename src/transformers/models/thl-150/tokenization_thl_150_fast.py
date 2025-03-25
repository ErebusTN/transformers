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
from pathlib import Path
from typing import Optional, Tuple, Union
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_thl_150 import THL150Tokenizer

# Set up logging
logger = logging.get_logger(__name__)

class TokenizationError(ValueError):
    """Base class for tokenization-related errors with contextual information."""
    def __init__(self, message: str, component: str, value: any = None):
        full_msg = f"Tokenization error in {component}: {message}"
        if value is not None:
            full_msg += f" (received {type(value).__name__} {value})"
        super().__init__(full_msg)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

MAX_MODEL_INPUT_SIZES = {"thl/thl-150-tokenizer": 32768}

class THL150TokenizerFast(PreTrainedTokenizerFast):
    """
    Optimized THL-150 tokenizer with comprehensive validation and error handling.
    
    Implements byte-level BPE with enhanced safety checks and detailed error reporting.
    
    Example:
    ```python
    tokenizer = THL150TokenizerFast.from_pretrained("thl/thl-150-tokenizer")
    encoded = tokenizer.encode("Hello THL-150!", return_tensors="pt")
    ```
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = THL150Tokenizer
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES

    def __init__(
        self,
        vocab_file: Optional[Union[str, Path]] = None,
        merges_file: Optional[Union[str, Path]] = None,
        tokenizer_file: Optional[Union[str, Path]] = None,
        unk_token: Union[str, AddedToken] = "<|endoftext|>",
        bos_token: Optional[Union[str, AddedToken]] = None,
        eos_token: Union[str, AddedToken] = "<|endoftext|>",
        pad_token: Union[str, AddedToken] = "<|endoftext|>",
        **kwargs,
    ):
        """Initialize tokenizer with rigorous file validation and token checks."""
        try:
            # --------------------------
            # File Validation
            # --------------------------
            if not tokenizer_file and (not vocab_file or not merges_file):
                raise TokenizationError(
                    "Must provide either tokenizer_file or both vocab_file and merges_file",
                    "file_validation"
                )

            # --------------------------
            # Token Validation
            # --------------------------
            def validate_token(token: Union[str, AddedToken], name: str) -> AddedToken:
                """Validate and normalize special tokens."""
                if isinstance(token, str):
                    if not token:
                        raise TokenizationError(
                            f"{name} cannot be empty string", 
                            "token_validation",
                            token
                        )
                    return AddedToken(token, special=True)
                if not isinstance(token, AddedToken):
                    raise TokenizationError(
                        f"{name} must be str or AddedToken", 
                        "token_validation",
                        type(token)
                    )
                return token

            bos_token = validate_token(bos_token, "bos_token") if bos_token else None
            eos_token = validate_token(eos_token, "eos_token")
            unk_token = validate_token(unk_token, "unk_token")
            pad_token = validate_token(pad_token, "pad_token")

            # --------------------------
            # Encoding Validation
            # --------------------------
            model_max_length = kwargs.get("model_max_length", self.max_model_input_sizes)
            if model_max_length > MAX_MODEL_INPUT_SIZES["thl/thl-150-tokenizer"]:
                raise TokenizationError(
                    f"model_max_length exceeds maximum allowed size {MAX_MODEL_INPUT_SIZES}",
                    "length_validation",
                    model_max_length
                )

            # --------------------------
            # Parent Initialization
            # --------------------------
            super().__init__(
                vocab_file=vocab_file,
                merges_file=merges_file,
                tokenizer_file=tokenizer_file,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                **kwargs,
            )

            # --------------------------
            # Post-Init Validation
            # --------------------------
            if len(self.vocab) < 100:
                logger.warning("Vocab size seems unusually small (%d tokens)", len(self.vocab))

        except TokenizationError as te:
            logger.error("Tokenizer initialization failed: %s", str(te))
            raise
        except Exception as e:
            logger.critical("Unexpected tokenizer error: %s", str(e))
            raise RuntimeError("Failed to initialize tokenizer") from e

    def save_vocabulary(
        self, 
        save_directory: Union[str, Path],
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save tokenizer files with path validation and error handling.
        
        Args:
            save_directory: Path to save directory
            filename_prefix: Optional prefix for filenames
            
        Returns:
            Tuple of saved file paths
            
        Raises:
            TokenizationError: If save operation fails
        """
        try:
            save_path = Path(save_directory)
            if not save_path.exists():
                save_path.mkdir(parents=True)
            elif not save_path.is_dir():
                raise TokenizationError(
                    "Save path must be a directory", 
                    "save_vocabulary",
                    save_directory
                )

            return super().save_vocabulary(str(save_path), filename_prefix)
            
        except IOError as e:
            raise TokenizationError(
                f"Failed to save vocabulary: {str(e)}",
                "save_vocabulary"
            ) from e
        except Exception as e:
            logger.error("Vocabulary save failed: %s", str(e))
            raise

    def _validate_text_input(self, text: str):
        """Validate input text before processing."""
        if not isinstance(text, str):
            raise TokenizationError(
                "Input must be a string", 
                "text_validation",
                type(text)
            )
        if not text.strip():
            raise TokenizationError(
                "Input text cannot be empty", 
                "text_validation"
            )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs
    ):
        """Enhanced encode method with input validation."""
        self._validate_text_input(text)
        return super().encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        **kwargs
    ):
        """Safe decode method with ID validation."""
        try:
            return super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        except Exception as e:
            raise TokenizationError(
                f"Decoding failed: {str(e)}",
                "decode_operation",
                token_ids
            ) from e