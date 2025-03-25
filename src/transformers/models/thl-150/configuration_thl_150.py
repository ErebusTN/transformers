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

"""THL-150 model configuration"""

from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging

logger = logging.get_logger(__name__)

class ConfigurationError(ValueError):
    """Base class for configuration-related errors with enhanced formatting."""
    def __init__(self, param: str, message: str, value: any):
        full_msg = f"Invalid configuration for {param}: {message} (received {type(value).__name__} {value})"
        super().__init__(full_msg)

def validate_positive(param: str, value: Union[int, float], strict_int: bool = False) -> None:
    """
    Validate that a value is a positive number.
    
    Args:
        param: Name of the parameter being validated
        value: Value to check
        strict_int: Require integer type if True
    """
    if strict_int and not isinstance(value, int):
        raise ConfigurationError(param, "must be a positive integer", value)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(param, "must be a positive number", value)

def validate_choice(param: str, value: any, valid_options: set) -> None:
    """
    Validate that a value is in a set of allowed options.
    
    Args:
        param: Name of the parameter being validated
        value: Value to check
        valid_options: Set of allowed values
    """
    if value not in valid_options:
        raise ConfigurationError(
            param,
            f"must be one of {sorted(valid_options)}",
            value
        )

class THL150Config(PretrainedConfig):
    """
    Configuration class for THL-150 model with enhanced validation and error handling.
    
    This class ensures all model parameters are within valid ranges and properly configured
    before model instantiation. It provides detailed error messages to help identify
    configuration issues quickly.
    """
    model_type = "thl_150"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # Default values act as both fallbacks and documentation
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initialize THL-150 configuration with rigorous parameter validation.
        
        Each parameter is validated for:
        - Type correctness
        - Value validity
        - Inter-parameter consistency
        """
        try:
            # --------------------------
            # Basic Type Validation
            # --------------------------
            # Validate fundamental parameter types
            validate_positive("vocab_size", vocab_size, strict_int=True)
            validate_positive("hidden_size", hidden_size, strict_int=True)
            validate_positive("intermediate_size", intermediate_size, strict_int=True)
            validate_positive("num_hidden_layers", num_hidden_layers, strict_int=True)
            validate_positive("num_attention_heads", num_attention_heads, strict_int=True)
            validate_positive("max_position_embeddings", max_position_embeddings, strict_int=True)
            validate_positive("initializer_range", initializer_range)
            validate_positive("rms_norm_eps", rms_norm_eps)
            validate_positive("rope_theta", rope_theta)

            # --------------------------
            # Attention Head Validation
            # --------------------------
            # Set default and validate key/value heads
            num_key_value_heads = num_key_value_heads or num_attention_heads
            validate_positive("num_key_value_heads", num_key_value_heads, strict_int=True)
            
            # Ensure compatible attention head configuration
            if num_key_value_heads > num_attention_heads:
                raise ConfigurationError(
                    "num_key_value_heads",
                    f"cannot exceed num_attention_heads ({num_attention_heads})",
                    num_key_value_heads
                )
                
            if num_attention_heads % num_key_value_heads != 0:
                raise ConfigurationError(
                    "num_attention_heads",
                    f"must be divisible by num_key_value_heads ({num_key_value_heads})",
                    num_attention_heads
                )

            # --------------------------
            # Dimension Compatibility
            # --------------------------
            # Hidden size must be divisible by attention heads
            if hidden_size % num_attention_heads != 0:
                raise ConfigurationError(
                    "hidden_size",
                    f"must be divisible by num_attention_heads ({num_attention_heads})",
                    hidden_size
                )

            # --------------------------
            # Activation Function Check
            # --------------------------
            validate_choice(
                "hidden_act", 
                hidden_act, 
                {"silu", "gelu", "relu", "gelu_new"}
            )

            # --------------------------
            # Sliding Window Validation
            # --------------------------
            if not isinstance(use_sliding_window, bool):
                raise ConfigurationError(
                    "use_sliding_window",
                    "must be a boolean value",
                    use_sliding_window
                )
                
            if use_sliding_window:
                validate_positive("sliding_window", sliding_window, strict_int=True)
                
                if sliding_window > max_position_embeddings:
                    raise ConfigurationError(
                        "sliding_window",
                        f"cannot exceed max_position_embeddings ({max_position_embeddings})",
                        sliding_window
                    )
                    
                if max_window_layers < 0 or max_window_layers > num_hidden_layers:
                    raise ConfigurationError(
                        "max_window_layers",
                        f"must be between 0 and num_hidden_layers ({num_hidden_layers})",
                        max_window_layers
                    )

            # --------------------------
            # RoPE Scaling Configuration
            # --------------------------
            if rope_scaling is not None:
                if not isinstance(rope_scaling, dict):
                    raise ConfigurationError(
                        "rope_scaling",
                        "must be a dictionary",
                        rope_scaling
                    )
                    
                # Support both 'type' and 'rope_type' keys
                if 'type' in rope_scaling:
                    rope_scaling = rope_scaling.copy()
                    rope_scaling['rope_type'] = rope_scaling.pop('type')
                    
                if 'rope_type' not in rope_scaling:
                    raise ConfigurationError(
                        "rope_scaling",
                        "must contain 'type' key specifying scaling strategy",
                        rope_scaling
                    )
                    
                if 'factor' not in rope_scaling:
                    raise ConfigurationError(
                        "rope_scaling",
                        "must contain 'factor' key specifying scaling factor",
                        rope_scaling
                    )
                    
                validate_positive("rope_scaling.factor", rope_scaling['factor'])

            # --------------------------
            # Dropout Validation
            # --------------------------
            if not (0 <= attention_dropout <= 1):
                raise ConfigurationError(
                    "attention_dropout",
                    "must be between 0 and 1 (inclusive)",
                    attention_dropout
                )

            # --------------------------
            # Final Assignment
            # --------------------------
            # Only assign values after successful validation
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.use_cache = use_cache
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.use_sliding_window = use_sliding_window
            self.sliding_window = sliding_window
            self.max_window_layers = max_window_layers
            self.attention_dropout = attention_dropout

            # --------------------------
            # External RoPE Validation
            # --------------------------
            try:
                rope_config_validation(self)
            except ValueError as e:
                raise ConfigurationError(
                    "rope_config", 
                    f"external validation failed: {str(e)}", 
                    self.rope_scaling
                ) from e

            # Initialize parent class after local validation
            super().__init__(
                tie_word_embeddings=tie_word_embeddings,
                **kwargs
            )

        except ConfigurationError as ce:
            logger.error("Configuration validation failed: %s", str(ce))
            raise
        except Exception as e:
            logger.critical("Unexpected configuration error: %s", str(e))
            raise RuntimeError("Failed to initialize model configuration") from e