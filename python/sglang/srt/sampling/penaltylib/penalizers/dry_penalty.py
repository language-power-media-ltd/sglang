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
        # 保存输入和输出令牌序列（每个 batch 元素皆为一个 1D torch.Tensor）
        self.input_token_ids: List[torch.Tensor] = None
        self.output_token_ids: List[torch.Tensor] = None

        # DRY 参数张量，均为 batch 维：shape = (batch_size, ...) 
        self.dry_multipliers: torch.Tensor = None         # float, shape (B, 1)
        self.dry_bases: torch.Tensor = None               # float, shape (B, 1)
        self.dry_allowed_lengths: torch.Tensor = None       # int, shape (B,)
        self.dry_ranges: torch.Tensor = None              # int, shape (B,)
        self.dry_max_ngram: torch.Tensor = None           # int, shape (B,)
        self.dry_max_occurrences: torch.Tensor = None     # int, shape (B,)
        self.dry_early_exit_match_len: torch.Tensor = None  # int, shape (B,)
        # 每个 batch 对应的序列中断 token（列表形式），如果 token 出现在末尾，则不惩罚
        self.dry_sequence_breakers: List[List[int]] = None

    def _is_required(self) -> bool:
        # 当存在一个生成请求的 dry_multiplier 不为 0 时，启用 DRY 惩罚
        return any(
            req.sampling_params.dry_multiplier != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        for req in self.orchestrator.reqs():
            print(req.sampling_params.dry_allowed_length)
            for prompt in req.sampling_params.dry_sequence_breakers:
                print(req.tokenizer.encode(prompt, add_special_tokens=False))
        # 提取每个请求的 DRY 参数
        dry_multipliers = [req.sampling_params.dry_multiplier for req in self.orchestrator.reqs()]
        dry_bases = [req.sampling_params.dry_base for req in self.orchestrator.reqs()]
        dry_allowed_lengths = [req.sampling_params.dry_allowed_length for req in self.orchestrator.reqs()]
        dry_ranges = [req.sampling_params.dry_range for req in self.orchestrator.reqs()]
        dry_max_ngram = [req.sampling_params.dry_max_ngram for req in self.orchestrator.reqs()]
        dry_max_occurrences = [req.sampling_params.dry_max_occurrences for req in self.orchestrator.reqs()]
        dry_early_exit_match_len = [req.sampling_params.dry_early_exit_match_len for req in self.orchestrator.reqs()]
        dry_sequence_breakers  = [
            [req.tokenizer.encode(prompt, add_special_tokens=False)[-1] 
            for prompt in req.sampling_params.dry_sequence_breakers]
            for req in self.orchestrator.reqs()
        ]

        device = self.orchestrator.device
        self.dry_multipliers = torch.tensor(dry_multipliers, dtype=torch.float32, device=device).unsqueeze(1)
        self.dry_bases = torch.tensor(dry_bases, dtype=torch.float32, device=device).unsqueeze(1)
        self.dry_allowed_lengths = torch.tensor(dry_allowed_lengths, dtype=torch.int32, device=device)
        self.dry_ranges = torch.tensor(dry_ranges, dtype=torch.int32, device=device)
        self.dry_max_ngram = torch.tensor(dry_max_ngram, dtype=torch.int32, device=device)
        self.dry_max_occurrences = torch.tensor(dry_max_occurrences, dtype=torch.int32, device=device)
        self.dry_early_exit_match_len = torch.tensor(dry_early_exit_match_len, dtype=torch.int32, device=device)
        self.dry_sequence_breakers = dry_sequence_breakers

    def _teardown(self):
        self.dry_multipliers = None
        self.dry_bases = None
        self.dry_allowed_lengths = None
        self.dry_ranges = None
        self.dry_max_ngram = None
        self.dry_max_occurrences = None
        self.dry_early_exit_match_len = None
        self.dry_sequence_breakers = None
        self.input_token_ids = None
        self.output_token_ids = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        # 假设 input_ids.token_ids 是一个列表，每个元素为对应 batch 的 1D tensor
        self.input_token_ids = input_ids.token_ids

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        # 如果 output_ids.token_ids 已经是列表，就直接存储；否则拆分为列表（每个 batch 一行）
        if isinstance(output_ids.token_ids, list):
            new_output_tokens = output_ids.token_ids
        else:
            new_output_tokens = [output_ids.token_ids[i] for i in range(output_ids.token_ids.size(0))]
        
        # 保存新生成的输出 token
        self.output_token_ids = new_output_tokens

        # 将新的输出 token 追加到现有的 input token 序列中
        if self.input_token_ids is not None:
            for i in range(len(new_output_tokens)):
                # 保证 self.input_token_ids[i] 至少为 1D tensor
                if self.input_token_ids[i].dim() == 0:
                    self.input_token_ids[i] = self.input_token_ids[i].unsqueeze(0)
                # 保证 new_output_tokens[i] 也是 1D tensor
                if new_output_tokens[i].dim() == 0:
                    new_output_tokens[i] = new_output_tokens[i].unsqueeze(0)
                self.input_token_ids[i] = torch.cat([self.input_token_ids[i], new_output_tokens[i]])
        else:
            self.input_token_ids = new_output_tokens

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        vocab_size = self.orchestrator.vocab_size

        for i in range(batch_size):
            # 处理输入 token 序列
            if self.input_token_ids is not None:
                input_seq = self.input_token_ids[i]
                if input_seq.dim() == 0:
                    input_seq = input_seq.unsqueeze(0)
                prompt_len = input_seq.size(0) - (input_seq == vocab_size).sum().item()
            else:
                prompt_len = 0
            print("输入"+str(prompt_len))
            # 处理输出 token 序列
            if self.output_token_ids is not None and i < len(self.output_token_ids):
                output_seq = self.output_token_ids[i]
                if output_seq.dim() == 0:
                    output_seq = output_seq.unsqueeze(0)
                output_len = output_seq.size(0) - (output_seq == vocab_size).sum().item()
            else:
                output_len = 0
            print("输入"+str(output_len))
            # 获取实际的 token 序列（排除填充部分）
            seq_prompt = input_seq[:prompt_len] if prompt_len > 0 else torch.tensor([], dtype=torch.int64, device=logits.device)
            seq_output = output_seq[:output_len] if output_len > 0 else torch.tensor([], dtype=torch.int64, device=logits.device)
            # 拼接完整的 token 序列
            if seq_prompt.numel() > 0 and seq_output.numel() > 0:
                token_seq = torch.cat((seq_prompt, seq_output), dim=0)
            elif seq_prompt.numel() > 0:
                token_seq = seq_prompt
            elif seq_output.numel() > 0:
                token_seq = seq_output
            else:
                print("结束1")
                continue
            print("总共"+str(token_seq))
            # 以下逻辑保持不变……
            # 仅考虑最近 dry_range 个 token
            range_limit = self.dry_ranges[i].item()
            if range_limit > 0 and token_seq.size(0) > range_limit:
                token_seq = token_seq[-range_limit:]

            if token_seq.size(0) < 2:
                print("结束2")
                continue

            last_token = token_seq[-1].item()
            if last_token in self.dry_sequence_breakers[i]:
                continue

            break_mask = torch.zeros(token_seq.size(0), dtype=torch.bool, device=logits.device)
            for breaker in self.dry_sequence_breakers[i]:
                break_mask.logical_or_(token_seq == breaker)

            max_ngram_val = self.dry_max_ngram[i].item()
            curr_max_ngram = 0
            for curr_max_ngram in range(min(len(break_mask), max_ngram_val + 1)):
                if break_mask[-curr_max_ngram - 1]:
                    break

            min_ngram = self.dry_allowed_lengths[i].item()
            if curr_max_ngram <= min_ngram:
                print("结束3")
                continue

            ngram_lens = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
            endpoints_all = torch.nonzero(token_seq == last_token, as_tuple=True)[0].tolist()
            if len(endpoints_all) < 2:
                print("结束4")
                continue
            endpoint_indexes = endpoints_all[:-1]
            max_occ_val = self.dry_max_occurrences[i].item()
            if len(endpoint_indexes) > max_occ_val:
                endpoint_indexes = endpoint_indexes[-max_occ_val:]
            early_exit_match_len_val = self.dry_early_exit_match_len[i].item()

            for idx in reversed(endpoint_indexes):
                if idx == token_seq.size(0) - 1:
                    print("结束5")
                    continue

                match_len = 0
                limit = min(idx, curr_max_ngram)
                for unwind in range(1, limit + 1):
                    if break_mask[idx - unwind]:
                        print("结束6")
                        break
                    if token_seq[idx - unwind] != token_seq[-unwind - 1]:
                        print("结束7")
                        break
                    match_len = unwind

                if match_len > 0:
                    next_tok = token_seq[idx + 1].item()
                    new_len = match_len + 1
                    ngram_lens[next_tok] = max(ngram_lens[next_tok].item(), new_len)
                    if new_len >= early_exit_match_len_val:
                        print("结束8")
                        break

            penalty_mask = ngram_lens > 0
            if penalty_mask.any():
                scales = self.dry_bases[i].item() ** (ngram_lens[penalty_mask] - min_ngram)
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
        self.dry_sequence_breakers = [self.dry_sequence_breakers[i] for i in indices_to_keep]
        if self.input_token_ids is not None:
            self.input_token_ids = [self.input_token_ids[i] for i in indices_to_keep]
        if self.output_token_ids is not None:
            self.output_token_ids = [self.output_token_ids[i] for i in indices_to_keep]

    def _merge(self, their: "BatchedDRYPenalizer"):
        # 合并其他 orchestrator 中 DRY 惩罚器的状态
        other = their  # 假设 other 也是 BatchedDRYPenalizer
        self.dry_multipliers = torch.cat([self.dry_multipliers, other.dry_multipliers], dim=0)
        self.dry_bases = torch.cat([self.dry_bases, other.dry_bases], dim=0)
        self.dry_allowed_lengths = torch.cat([self.dry_allowed_lengths, other.dry_allowed_lengths], dim=0)
        self.dry_ranges = torch.cat([self.dry_ranges, other.dry_ranges], dim=0)
        self.dry_max_ngram = torch.cat([self.dry_max_ngram, other.dry_max_ngram], dim=0)
        self.dry_max_occurrences = torch.cat([self.dry_max_occurrences, other.dry_max_occurrences], dim=0)
        self.dry_early_exit_match_len = torch.cat([self.dry_early_exit_match_len, other.dry_early_exit_match_len], dim=0)
        self.dry_sequence_breakers = self.dry_sequence_breakers + other.dry_sequence_breakers
        if self.input_token_ids is None:
            self.input_token_ids = other.input_token_ids
        else:
            self.input_token_ids.extend(other.input_token_ids)
        if self.output_token_ids is None:
            self.output_token_ids = other.output_token_ids
        else:
            self.output_token_ids.extend(other.output_token_ids)