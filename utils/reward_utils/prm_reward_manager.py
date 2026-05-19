"""
PRM Reward Manager for SC (Self-Correction) GRPO training.

Two reward placement strategies per step:
  - "Fail rubrics:" position  → char-to-token conversion (multi-token phrase)
  - action token position     → direct token-ID search (<|solve|> etc. are single special tokens)

Requires response decoded with skip_special_tokens=False so action tokens are visible as text.
"""

import asyncio
import re
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager

_ACTION_TOKEN_STRS = ["<|solve|>", "<|rethink|>", "<|end|>"]


def _extract_problem_from_prompt(prompt_str: str) -> str:
    """디코딩된 프롬프트에서 실제 수학 문제 텍스트만 추출.
    preprocess.py가 생성하는 포맷: '[Problem]\\n<문제>\\n\\nWrite Step N.'
    """
    m = re.search(r"\[Problem\]\s*\n(.*?)(?:\n\nWrite Step|\Z)", prompt_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: 마지막 user 메시지 전체
    m2 = re.search(r"user\s*\n(.*?)$", prompt_str, re.DOTALL)
    return m2.group(1).strip() if m2 else prompt_str.strip()


class PRMRewardManager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer,
        num_examine: int = 1,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self._print_counts: dict[str, int] = {}

        # Resolve action special-token IDs once
        self._action_ids: set[int] = set()
        for tok_str in _ACTION_TOKEN_STRS:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid != tokenizer.unk_token_id:
                self._action_ids.add(tid)

    def _char_to_token_pos(self, response_str: str, char_pos: int) -> int:
        """Token index (within response) for a char position. Uses full text including special tokens."""
        prefix = response_str[:char_pos]
        return len(self.tokenizer.encode(prefix, add_special_tokens=False))

    def _find_action_token_positions(self, response_ids: torch.Tensor, n_valid: int) -> list[int]:
        """Return token positions of each action special token in order."""
        positions = []
        for pos in range(n_valid):
            if response_ids[pos].item() in self._action_ids:
                positions.append(pos)
        return positions

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm is not None:
            return reward_from_rm

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            # skip_special_tokens=False so <|solve|> etc. appear in text for reward_func
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(result, dict) and "rubric_rewards" in result:
                # --- rubric rewards: char pos → token pos ---
                for reward_val, char_pos in result["rubric_rewards"]:
                    if reward_val == 0.0:
                        continue
                    tok_pos = min(self._char_to_token_pos(response_str, char_pos), valid_response_length - 1)
                    reward_tensor[i, tok_pos] += reward_val

                # --- action rewards: direct token ID search ---
                action_positions = self._find_action_token_positions(valid_response_ids, valid_response_length)
                for j, reward_val in enumerate(result["action_rewards"]):
                    if reward_val == 0.0 or j >= len(action_positions):
                        continue
                    reward_tensor[i, action_positions[j]] += reward_val

                score = result.get("score", 0.0)
            else:
                score = float(result) if not isinstance(result, dict) else result.get("score", 0.0)
                reward_tensor[i, valid_response_length - 1] = score

            reward_extra_info["score"].append(score)

            if data_source not in self._print_counts:
                self._print_counts[data_source] = 0
            if self._print_counts[data_source] < self.num_examine:
                self._print_counts[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    async def run_single(self, data) -> dict:
        """Async interface for the experimental RewardLoopWorker path."""
        assert len(data) == 1
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        loop = asyncio.get_event_loop()
        response_str = await loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids.tolist(), skip_special_tokens=False)
        )

        data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})

        # extra_info에 problem이 없으면 prompt에서 실제 수학 문제 부분만 추출해 fallback
        if "problem" not in extra_info:
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = await loop.run_in_executor(
                None, lambda: self.tokenizer.decode(valid_prompt_ids.tolist(), skip_special_tokens=True)
            )
            extra_info["problem"] = _extract_problem_from_prompt(prompt_str)

        result = await loop.run_in_executor(
            None,
            lambda: self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
        )

        if isinstance(result, dict) and "rubric_rewards" in result:
            step_reward_positions = []

            for reward_val, char_pos in result["rubric_rewards"]:
                if reward_val != 0.0:
                    tok_pos = min(self._char_to_token_pos(response_str, char_pos), valid_response_length - 1)
                    step_reward_positions.append((tok_pos, reward_val))

            action_positions = self._find_action_token_positions(valid_response_ids, valid_response_length)
            for j, reward_val in enumerate(result["action_rewards"]):
                if j < len(action_positions) and reward_val != 0.0:
                    step_reward_positions.append((action_positions[j], reward_val))

            score = result.get("score", 0.0)
            return {
                "reward_score": score,
                "step_reward_positions": step_reward_positions,
                "reward_extra_info": {"score": score},
            }
        else:
            score = float(result) if not isinstance(result, dict) else result.get("score", 0.0)
            return {"reward_score": score, "reward_extra_info": {"score": score}}
