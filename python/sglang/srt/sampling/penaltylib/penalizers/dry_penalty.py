from typing import List
import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedDRYPenalizer(_BatchedPenalizer):
    """
    使用 Don't Repeat Yourself (DRY) 采样惩罚，防止输出中出现“循环”。

    惩罚计算公式为：

        penalty = dry_multiplier * dry_base^(n - dry_allowed_length)

    其中 n 表示当前 token 前与历史末尾匹配的连续 token 个数，
    当 n 小于 dry_allowed_length 时不惩罚。

    参数均通过 req.sampling_params 中对应的 dry_* 字段传入。
    """
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        # 保存输入和输出令牌序列（每个 batch 元素皆为 1D torch.Tensor）
        self.input_token_ids: List[torch.Tensor] = None       # 初始 prompt tokens
        self.output_token_ids: List[torch.Tensor] = None      # 当前生成的最新 output tokens
        self.cumulated_output_token_ids: List[torch.Tensor] = []  # 已生成的累计 output tokens
        self.cached_input_prompt: List[torch.Tensor] = []     # 缓存已处理的 prompt tokens

        # DRY 参数（每个参数均为 batch 维张量）
        self.dry_multipliers: torch.Tensor = None         # float, shape (B, 1)
        self.dry_bases: torch.Tensor = None               # float, shape (B, 1)
        self.dry_allowed_lengths: torch.Tensor = None     # int, shape (B,)
        self.dry_ranges: torch.Tensor = None              # int, shape (B,)
        self.dry_max_ngram: torch.Tensor = None           # int, shape (B,)
        self.dry_max_occurrences: torch.Tensor = None     # int, shape (B,)
        self.dry_early_exit_match_len: torch.Tensor = None  # int, shape (B,)
        # 每个 batch 对应的序列中断 token（列表形式），在末尾出现时不惩罚
        self.dry_sequence_breakers: List[List[int]] = []

    def _is_required(self) -> bool:
        # 当至少一个请求的 dry_multiplier 不为 0 时启用 DRY 惩罚
        reqs = self.orchestrator.reqs()
        return any(req.sampling_params.dry_multiplier != 0.0 for req in reqs)

    def _prepare(self):
        # 预处理 DRY 参数，避免多次调用 self.orchestrator.reqs()
        reqs = list(self.orchestrator.reqs())

        dry_multipliers = [req.sampling_params.dry_multiplier for req in reqs]
        dry_bases = [req.sampling_params.dry_base for req in reqs]
        dry_allowed_lengths = [req.sampling_params.dry_allowed_length for req in reqs]
        dry_ranges = [req.sampling_params.dry_range for req in reqs]
        dry_max_ngram = [req.sampling_params.dry_max_ngram for req in reqs]
        dry_max_occurrences = [req.sampling_params.dry_max_occurrences for req in reqs]
        dry_early_exit_match_len = [req.sampling_params.dry_early_exit_match_len for req in reqs]

        # 注意：dry_sequence_breakers 为列表中每个元素也是列表
        self.dry_sequence_breakers = [
            [req.tokenizer.encode(prompt, add_special_tokens=False)[-1]
             for prompt in req.sampling_params.dry_sequence_breakers]
            for req in reqs
        ]

        device = self.orchestrator.device
        # 构造 tensor，并确保 dry_multipliers 和 dry_bases 拥有 shape (B, 1)
        self.dry_multipliers = torch.tensor(
            dry_multipliers, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.dry_bases = torch.tensor(
            dry_bases, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.dry_allowed_lengths = torch.tensor(
            dry_allowed_lengths, dtype=torch.int32, device=device
        )
        self.dry_ranges = torch.tensor(
            dry_ranges, dtype=torch.int32, device=device
        )
        self.dry_max_ngram = torch.tensor(
            dry_max_ngram, dtype=torch.int32, device=device
        )
        self.dry_max_occurrences = torch.tensor(
            dry_max_occurrences, dtype=torch.int32, device=device
        )
        self.dry_early_exit_match_len = torch.tensor(
            dry_early_exit_match_len, dtype=torch.int32, device=device
        )

    def _teardown(self):
        self.dry_multipliers = None
        self.dry_bases = None
        self.dry_allowed_lengths = None
        self.dry_ranges = None
        self.dry_max_ngram = None
        self.dry_max_occurrences = None
        self.dry_early_exit_match_len = None
        self.dry_sequence_breakers = []
        self.input_token_ids = None
        self.output_token_ids = None
        self.cumulated_output_token_ids = []
        self.cached_input_prompt = []

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        # 过滤掉特殊 token（假设 tokenizer 有 bos_token_id 和 eos_token_id 属性）
        req = self.orchestrator.reqs()[0]
        bos_id = req.tokenizer.bos_token_id
        eos_id = req.tokenizer.eos_token_id

        self.input_token_ids = []
        self.cached_input_prompt = []
        for tokens in input_ids.token_ids:
            filtered = tokens[(tokens != bos_id) & (tokens != eos_id)]
            self.input_token_ids.append(filtered)
            self.cached_input_prompt.append(filtered)

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        if isinstance(output_ids.token_ids, list):
            new_output_tokens = output_ids.token_ids
        else:
            new_output_tokens = [output_ids.token_ids[i] for i in range(output_ids.token_ids.size(0))]
        
        # 初始化 cumulated_output_token_ids
        if self.cumulated_output_token_ids is None or len(self.cumulated_output_token_ids) == 0:
            self.cumulated_output_token_ids = [None] * len(new_output_tokens)
        if len(self.cumulated_output_token_ids) < len(new_output_tokens):
            self.cumulated_output_token_ids.extend(
                [None] * (len(new_output_tokens) - len(self.cumulated_output_token_ids))
            )
        
        for i in range(len(new_output_tokens)):
            out_tok = new_output_tokens[i]
            if self.cumulated_output_token_ids[i] is None:
                self.cumulated_output_token_ids[i] = out_tok.unsqueeze(0) if out_tok.dim() == 0 else out_tok
            else:
                if self.cumulated_output_token_ids[i].dim() == 0:
                    self.cumulated_output_token_ids[i] = self.cumulated_output_token_ids[i].unsqueeze(0)
                if out_tok.dim() == 0:
                    out_tok = out_tok.unsqueeze(0)
                self.cumulated_output_token_ids[i] = torch.cat(
                    [self.cumulated_output_token_ids[i], out_tok]
                )
        
        self.output_token_ids = new_output_tokens

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = len(self.orchestrator.reqs())
        vocab_size = logits.size(-1)
        device = logits.device

        # 确保 cached_input_prompt 数量与 batch_size 匹配
        if len(self.cached_input_prompt) < batch_size:
            self.cached_input_prompt += [torch.empty(0, dtype=torch.int64, device=device)] * (batch_size - len(self.cached_input_prompt))
        elif len(self.cached_input_prompt) > batch_size:
            self.cached_input_prompt = self.cached_input_prompt[:batch_size]

        for i in range(batch_size):
            if len(self.cached_input_prompt) == 0 or i >= len(self.cached_input_prompt):
                continue
            if self.dry_sequence_breakers is None:
                continue
            input_seq = self.cached_input_prompt[i] if self.cached_input_prompt is not None else torch.empty(0, dtype=torch.int64, device=device)
            output_seq = self.cumulated_output_token_ids[i] if (self.cumulated_output_token_ids is not None and i < len(self.cumulated_output_token_ids)) else torch.empty(0, dtype=torch.int64, device=device)
            
            prompt_len = input_seq.size(0) - (input_seq == vocab_size).sum().item() if input_seq.numel() > 0 else 0
            output_len = output_seq.size(0) - (output_seq == vocab_size).sum().item() if output_seq.numel() > 0 else 0

            seq_prompt = input_seq[:prompt_len] if prompt_len > 0 else torch.empty(0, dtype=torch.int64, device=device)
            seq_output = output_seq[:output_len] if output_len > 0 else torch.empty(0, dtype=torch.int64, device=device)
            token_seq = torch.cat((seq_prompt, seq_output), dim=0) if (seq_prompt.numel() or seq_output.numel()) else torch.empty(0, dtype=torch.int64, device=device)
            if token_seq.numel() < 2:
                continue

            last_token = token_seq[-1].item()
            if last_token in self.dry_sequence_breakers[i]:
                continue

            breaker_tensor = torch.tensor(self.dry_sequence_breakers[i], device=device)
            break_mask = torch.isin(token_seq, breaker_tensor)

            max_ngram_val = int(self.dry_max_ngram[i].item())
            min_ngram = int(self.dry_allowed_lengths[i].item())
            max_occ_val = int(self.dry_max_occurrences[i].item())
            early_exit_match_len_val = int(self.dry_early_exit_match_len[i].item())
            
            seq_length = token_seq.size(0)
            curr_max_ngram = 0
            for n in range(min(seq_length, max_ngram_val + 1)):
                if break_mask[seq_length - n - 1]:
                    break
                curr_max_ngram = n + 1

            if curr_max_ngram <= min_ngram:
                continue

            ngram_lens = torch.zeros(vocab_size, dtype=torch.int32, device=device)
            endpoints_all = (token_seq == last_token).nonzero(as_tuple=False).flatten().tolist()
            if len(endpoints_all) < 2:
                continue
            endpoint_indexes = endpoints_all[:-1]
            if len(endpoint_indexes) > max_occ_val:
                endpoint_indexes = endpoint_indexes[-max_occ_val:]

            for idx in reversed(endpoint_indexes):
                if idx == token_seq.size(0) - 1:
                    continue
                match_len = 0
                limit = min(idx, curr_max_ngram)
                for unwind in range(1, limit + 1):
                    if break_mask[idx - unwind]:
                        break
                    if token_seq[idx - unwind].item() != token_seq[-unwind - 1].item():
                        break
                    match_len = unwind
                if match_len > 0:
                    next_tok = token_seq[idx + 1].item()
                    new_len = match_len + 1
                    ngram_lens[next_tok] = max(int(ngram_lens[next_tok].item()), new_len)
                    if new_len >= early_exit_match_len_val:
                        break

            penalty_mask = ngram_lens > 0
            if penalty_mask.any():
                scales = self.dry_bases[i].item() ** (ngram_lens[penalty_mask].to(torch.float32) - min_ngram)
                logits[i][penalty_mask] -= self.dry_multipliers[i].item() * scales

        return logits

    def _filter(self, indices_to_keep: List[int], indices_tensor_to_keep: torch.Tensor):
        self.dry_multipliers = self.dry_multipliers[indices_tensor_to_keep]
        self.dry_bases = self.dry_bases[indices_tensor_to_keep]
        self.dry_allowed_lengths = self.dry_allowed_lengths[indices_tensor_to_keep]
        self.dry_ranges = self.dry_ranges[indices_tensor_to_keep]
        self.dry_max_ngram = self.dry_max_ngram[indices_tensor_to_keep]
        self.dry_max_occurrences = self.dry_max_occurrences[indices_tensor_to_keep]
        self.dry_early_exit_match_len = self.dry_early_exit_match_len[indices_tensor_to_keep]
        # 对列表按照给定索引过滤
        self.dry_sequence_breakers = [self.dry_sequence_breakers[i] for i in indices_to_keep]

    def _merge(self, their: "BatchedDRYPenalizer"):
        self.dry_multipliers = torch.cat(
            [self.dry_multipliers, their.dry_multipliers], dim=0
        )
        self.dry_bases = torch.cat(
            [self.dry_bases, their.dry_bases], dim=0
        )
        self.dry_allowed_lengths = torch.cat(
            [self.dry_allowed_lengths, their.dry_allowed_lengths], dim=0
        )
        self.dry_ranges = torch.cat(
            [self.dry_ranges, their.dry_ranges], dim=0
        )
        self.dry_max_ngram = torch.cat(
            [self.dry_max_ngram, their.dry_max_ngram], dim=0
        )
        self.dry_max_occurrences = torch.cat(
            [self.dry_max_occurrences, their.dry_max_occurrences], dim=0
        )
        self.dry_early_exit_match_len = torch.cat(
            [self.dry_early_exit_match_len, their.dry_early_exit_match_len], dim=0
        )
        # 直接合并两个列表，而非使用 .append()
        self.dry_sequence_breakers = self.dry_sequence_breakers + their.dry_sequence_breakers