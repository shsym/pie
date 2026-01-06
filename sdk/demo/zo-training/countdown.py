import re
from pathlib import Path
from typing import Any

import pandas as pd
from rewards import format_reward, extract_answer


def format_problem(numbers: list[int], target: int) -> str:
    prompt = f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Please reason step by step."
    return prompt


class CountdownDataset:
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "data")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(
            {
                "problem": format_problem(item["nums"], item["target"]),
                "verifier": lambda response: verify_answer(
                    response, item["nums"], item["target"]
                ),
            }
        )
        return item


def answer_reward_function(
    response: str, numbers: list[int] = None, target: int = None
) -> float:
    answer_content = extract_answer(response)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.split(r"[\s()*/+-]+", answer_content) if n]

    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def verify_answer(
    response: str,
    numbers: list[int] = None,
    target: int = None,
) -> dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    answer_reward_value = answer_reward_function(response, numbers, target)
    format_reward_value = format_reward(response)

    total_reward = 0.1 * format_reward_value + answer_reward_value
    return {
        "reward": total_reward,
        "format_reward": format_reward_value,
        "answer_reward": answer_reward_value,
    }
