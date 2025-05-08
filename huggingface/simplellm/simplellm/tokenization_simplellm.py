# Copyright 2025 (c) Alex Huo
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""Tokenization classes for SimpleLLM."""

import json
from typing import TYPE_CHECKING, List
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.tokenization_utils import logging

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

class SimpleTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        pad_token=None,
        special_tokens=SPECIAL_TOKENS,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=True,
        add_prefix_space=True,
        **kwargs,
    ):
        # Wrap special tokens with AddedToken
        wrapped_special_tokens = [AddedToken(tok, normalized=False, special=True) for tok in special_tokens]

        # Core vocab
        self.word_to_id = {
            "I": 0, "You": 1, "like": 2, "do": 3, "not": 4, "coffee": 5,
            "tea": 6, ".": 7
        }

        # Add special tokens to vocab
        current_index = len(self.word_to_id)
        for tok in special_tokens:
            if tok not in self.word_to_id:
                self.word_to_id[tok] = current_index
                current_index += 1

        # Add pad_token if defined
        if pad_token and pad_token not in self.word_to_id:
            self.word_to_id[pad_token] = current_index

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab = self.word_to_id.copy()

        self.vocab_file = vocab_file
        self.add_prefix_space = add_prefix_space
        self.init_inputs = (vocab_file,)

        super().__init__(
            pad_token=pad_token,
            additional_special_tokens=special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_special_tokens({"additional_special_tokens": wrapped_special_tokens})

    def get_vocab(self):
        return self.vocab

    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        if len(text) == 0:
            return super().tokenize(text, **kwargs)

        if self.add_prefix_space:
            text = " " + text

        tokens = super().tokenize(text, **kwargs)

        if len(tokens) > 1 and tokens[0] == " " and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    def _tokenize(self, text, **kwargs):
        # Add spaces around special tokens to ensure they’re preserved as units
        for special in self.all_special_tokens:
            text = text.replace(special, f" {special} ")
        return text.strip().split()

    def _convert_token_to_id(self, token):
        return self.word_to_id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.id_to_word.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = f"{save_directory}/vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_tokenizer_config(self):
        return {
            "do_lower_case": False,
            "clean_up_tokenization_spaces": self.clean_up_tokenization_spaces,
            "add_prefix_space": self.add_prefix_space,
            "pad_token": self.pad_token,
            "additional_special_tokens": self.additional_special_tokens,
        }
