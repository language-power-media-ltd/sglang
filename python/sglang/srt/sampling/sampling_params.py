# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Sampling parameters for text generation."""

from typing import Any, Dict, List, Optional, Union

_SAMPLING_EPS = 1e-6


class SamplingParams:
    """
    The sampling parameters.

    See docs/references/sampling_params.md or
    https://docs.sglang.ai/references/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        spaces_between_special_tokens: bool = True,
        n: int = 1,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        no_stop_trim: bool = False,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        custom_params: Optional[Dict[str, Any]] = None,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_sequence_breakers: List[str] = [],
        dry_range: int = 0,
        dry_max_ngram: int = 12,
        dry_max_occurrences: int = 8,
        dry_early_exit_match_len: int = 8
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.stop_strs = stop
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.regex = regex
        self.n = n
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.no_stop_trim = no_stop_trim
        self.custom_params = custom_params
        self.dry_multiplier = dry_multiplier
        self.dry_base = dry_base
        self.dry_allowed_length = dry_allowed_length
        self.dry_range = dry_range
        self.dry_max_ngram = dry_max_ngram
        self.dry_max_occurrences = dry_max_occurrences
        self.dry_early_exit_match_len = dry_early_exit_match_len
        self.dry_sequence_breakers = dry_sequence_breakers

        # Process some special cases
        if self.temperature < _SAMPLING_EPS:
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = 1 << 30  # whole vocabulary

    def verify(self):
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in (0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in (0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in (0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.dry_multiplier < 0.0:
            raise ValueError(
                "dry_multiplier must be non-negative, got "
                f"{self.dry_multiplier}.")
        if self.dry_base <= 1.0:
            raise ValueError(
                "dry_base must be greater than 1, got "
                f"{self.dry_base}.")
        if self.dry_allowed_length < 0:
            raise ValueError(
                "dry_allowed_length must be non-negative, got "
                f"{self.dry_allowed_length}.")
        if self.dry_range < 0:
            raise ValueError(
                "dry_range must be non-negative, got "
                f"{self.dry_range}.")
        if self.dry_max_ngram < 0:
            raise ValueError(
                "dry_max_ngram must be non-negative, got "
                f"{self.dry_max_ngram}.")
        if self.dry_max_occurrences < 0:
            raise ValueError(
                "dry_max_occurrences must be non-negative, got "
                f"{self.dry_max_occurrences}.")
        if self.dry_early_exit_match_len < 0:
            raise ValueError(
                "dry_early_exit_match_len must be non-negative, got "
                f"{self.dry_early_exit_match_len}.")
        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def normalize(self, tokenizer):
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len
