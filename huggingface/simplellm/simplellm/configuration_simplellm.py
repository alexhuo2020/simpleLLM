# Copyright 2025 (c) Alex Huo
#
# This code is based on the HuggingFace's configuration_mixtral.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""SimpleLLM model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class SimpleLLMConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a [`SimpleLLM`]. It is used to instantiate an
    SimpleLLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the SimpleLLM.

    [SimpleLLM](https://huggingface.co/alex2020/simplellm) containing weights

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 8):
            Vocabulary size of the Grok model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GrokModel`]
        hidden_size (`int`, *optional*, defaults to 2):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 32):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder.
    """
    model_type = "simplellm"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
            self,
            vocab_size=int(9),
            hidden_size=int(2),
            num_hidden_layers=int(2),
            intermediate_size=int(32),
            hidden_act="relu",
            tie_word_embeddings=True,
            initializer_range=0.02,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


