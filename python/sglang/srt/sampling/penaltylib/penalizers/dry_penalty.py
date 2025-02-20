from typing import List, Optional
import torch
from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer, _TokenIDs
from sglang.srt.utils import get_compiler_backend

class BatchedDRYPenalizer(_BatchedPenalizer):
    """
    DRY penalizer applies penalties to discourage repeating n-grams in the output.
    """
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        self.multipliers: Optional[torch.Tensor] = None
        self.bases: Optional[torch.Tensor] = None
        self.allowed_lengths: Optional[torch.Tensor] = None
        self.ranges: Optional[torch.Tensor] = None
        self.max_ngram: Optional[torch.Tensor] = None
        self.max_occurrences: Optional[torch.Tensor] = None
        self.early_exit_match_len: Optional[torch.Tensor] = None
        self.sequence_breakers_ids: List[List[int]] = []
        self.input_token_seqs: List[torch.Tensor] = []
        self.output_token_seqs: List[torch.Tensor] = []

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.dry_multiplier != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        device = self.orchestrator.device

        # Collect DRY parameters from each request
        self.multipliers = torch.tensor(
            [req.sampling_params.dry_multiplier for req in reqs],
            dtype=torch.float32, device=device
        )
        self.bases = torch.tensor(
            [req.sampling_params.dry_base for req in reqs],
            dtype=torch.float32, device=device
        )
        self.allowed_lengths = torch.tensor(
            [req.sampling_params.dry_allowed_length for req in reqs],
            dtype=torch.int32, device=device
        )
        self.ranges = torch.tensor(
            [req.sampling_params.dry_range for req in reqs],
            dtype=torch.int32, device=device
        )
        self.max_ngram = torch.tensor(
            [req.sampling_params.dry_max_ngram for req in reqs],
            dtype=torch.int32, device=device
        )
        self.max_occurrences = torch.tensor(
            [req.sampling_params.dry_max_occurrences for req in reqs],
            dtype=torch.int32, device=device
        )
        self.early_exit_match_len = torch.tensor(
            [req.sampling_params.dry_early_exit_match_len for req in reqs],
            dtype=torch.int32, device=device
        )
        self.sequence_breakers_ids = [
            req.sampling_params.dry_sequence_breakers for req in reqs
        ]
        # Initialize token sequence storage
        self.input_token_seqs = []
        self.output_token_seqs = []

    def _teardown(self):
        # Cleanup parameters
        self.multipliers = None
        self.bases = None
        self.allowed_lengths = None
        self.ranges = None
        self.max_ngram = None
        self.max_occurrences = None
        self.early_exit_match_len = None
        self.sequence_breakers_ids = []
        self.input_token_seqs = []
        self.output_token_seqs = []

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        # Store input tokens for each request
        for irow in range(input_ids.batch_size()):
            seq_len = input_ids.seq_lens[irow]
            tokens = input_ids.token_ids[irow][:seq_len].clone().detach().to(self.orchestrator.device)
            if irow >= len(self.input_token_seqs):
                self.input_token_seqs.append(tokens)
            else:
                self.input_token_seqs[irow] = torch.cat([self.input_token_seqs[irow], tokens])

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        # Store output tokens for each request
        for irow in range(output_ids.batch_size()):
            seq_len = output_ids.seq_lens[irow]
            tokens = output_ids.token_ids[irow][:seq_len].clone().detach().to(self.orchestrator.device)
            if irow >= len(self.output_token_seqs):
                self.output_token_seqs.append(tokens)
            else:
                self.output_token_seqs[irow] = torch.cat([self.output_token_seqs[irow], tokens])

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.shape
        device = logits.device

        for irow in range(batch_size):
            if self.multipliers[irow] == 0:
                continue  # Skip if no penalty needed

            # Combine input and output tokens
            input_seq = self.input_token_seqs[irow].to(device)
            output_seq = self.output_token_seqs[irow].to(device)
            token_seq = torch.cat([input_seq, output_seq])

            # Apply range limitation
            range_limit = self.ranges[irow].item()
            if range_limit > 0:
                token_seq = token_seq[-range_limit:]

            if len(token_seq) < 2:
                continue  # Not enough context

            # Check sequence breakers
            last_token = token_seq[-1].item()
            if last_token in self.sequence_breakers_ids[irow]:
                continue

            # Build break mask
            break_mask = torch.zeros(len(token_seq), dtype=torch.bool, device=device)
            for break_tok in self.sequence_breakers_ids[irow]:
                break_mask |= (token_seq == break_tok)

            # Determine maximum n-gram length to check
            curr_max_ngram = 0
            max_ngram_val = self.max_ngram[irow].item()
            while curr_max_ngram < min(len(token_seq), max_ngram_val):
                if break_mask[-curr_max_ngram - 1]:
                    break
                curr_max_ngram += 1

            min_allowed = self.allowed_lengths[irow].item()
            if curr_max_ngram <= min_allowed:
                continue  # No penalty needed

            # Find previous occurrences
            ngram_lens = torch.zeros(vocab_size, dtype=torch.int32, device=device)
            endpoints = (token_seq == last_token).nonzero().view(-1).tolist()
            if len(endpoints) < 2:
                continue  # No previous occurrences

            # Limit occurrences to check
            max_occ = self.max_occurrences[irow].item()
            endpoints = endpoints[:-1][-max_occ:]  # Keep most recent occurrences

            # Check each occurrence for pattern matches
            early_exit_len = self.early_exit_match_len[irow].item()
            for idx in reversed(endpoints):
                match_len = 0
                for offset in range(1, min(idx, curr_max_ngram) + 1):
                    if break_mask[idx - offset] or token_seq[idx - offset] != token_seq[-offset - 1]:
                        break
                    match_len = offset

                if match_len > 0:
                    next_token = token_seq[idx + 1].item()
                    total_len = match_len + 1
                    if total_len > ngram_lens[next_token]:
                        ngram_lens[next_token] = total_len
                        if total_len >= early_exit_len:
                            break  # Early exit for this request

            # Apply penalties
            penalty_mask = ngram_lens > 0
            if penalty_mask.any():
                scales = self.bases[irow] ** (ngram_lens[penalty_mask] - min_allowed)
                logits[irow, penalty_mask] -= self.multipliers[irow] * scales

        return logits

    def _filter(self, indices_to_keep: List[int], indices_tensor_to_keep: torch.Tensor):
        # Filter stored data based on kept indices
        self.input_token_seqs = [self.input_token_seqs[i] for i in indices_to_keep]
        self.output_token_seqs = [self.output_token_seqs[i] for i in indices_to_keep]
        self.multipliers = self.multipliers[indices_tensor_to_keep]
        self.bases = self.bases[indices_tensor_to_keep]
        self.allowed_lengths = self.allowed_lengths[indices_tensor_to_keep]
        self.ranges = self.ranges[indices_tensor_to_keep]
        self.max_ngram = self.max_ngram[indices_tensor_to_keep]
        self.max_occurrences = self.max_occurrences[indices_tensor_to_keep]
        self.early_exit_match_len = self.early_exit_match_len[indices_tensor_to_keep]
        self.sequence_breakers_ids = [self.sequence_breakers_ids[i] for i in indices_to_keep]

    def _merge(self, their: "BatchedDRYPenalizer"):
        # Merge data from another penalizer instance
        self.input_token_seqs.extend(their.input_token_seqs)
        self.output_token_seqs.extend(their.output_token_seqs)
        self.multipliers = torch.cat([self.multipliers, their.multipliers])
        self.bases = torch.cat([self.bases, their.bases])
        self.allowed_lengths = torch.cat([self.allowed_lengths, their.allowed_lengths])
        self.ranges = torch.cat([self.ranges, their.ranges])
        self.max_ngram = torch.cat([self.max_ngram, their.max_ngram])
        self.max_occurrences = torch.cat([self.max_occurrences, their.max_occurrences])
        self.early_exit_match_len = torch.cat([self.early_exit_match_len, their.early_exit_match_len])
        self.sequence_breakers_ids.extend(their.sequence_breakers_ids)