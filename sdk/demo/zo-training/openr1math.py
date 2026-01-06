from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset
from rewards import extract_answer, format_reward, accuracy_reward
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

INSTRUCTION = "Please reason step by step, and put your final answer within \boxed{} (in the <answer> tag)."


class OpenR1MathDataset(Dataset):
    """Prepares the OpenR1-Math-220k dataset for training and testing."""

    def __init__(self, split: str = "train", test_size: int = 100):
        # Load the dataset from Hugging Face
        dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

        # Shuffle the dataset for an unbiased split
        shuffled_dataset = dataset.shuffle(seed=42)

        # Use .select() for proper splitting
        if split == "train":
            self.data = shuffled_dataset.select(
                range(len(shuffled_dataset) - test_size)
            )
        else:
            self.data = shuffled_dataset.select(
                range(len(shuffled_dataset) - test_size, len(shuffled_dataset))
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        return {
            "problem": item["problem"] + "\n" + INSTRUCTION,
            "answer": item["answer"],
            "verifier": lambda response: verify_answer(response, f"${item["answer"]}$"),
        }


def verify_answer(response: str, answer: str) -> dict[str, Any]:
    """
    Reward function for the OpenR1 Math dataset.
    Total reward = 0.1 * format_reward + answer_reward
    """
    attempted_answer = extract_answer(response)

    if attempted_answer is None:
        answer_reward_value = 0.0
    else:
        answer_reward_value = accuracy_reward(attempted_answer, answer)

    # If parsing or verification fails, accuracy reward is 0
    if answer_reward_value is None:
        answer_reward_value = 0.0

    format_reward_value = format_reward(response)

    # Use the corrected, weighted reward formula
    total_reward = 0.1 * format_reward_value + answer_reward_value

    return {
        "reward": total_reward,
        "format_reward": format_reward_value,
        "answer_reward": answer_reward_value,
    }


if __name__ == "__main__":
    print("Initializing test dataset (first 5 examples)...")
    # Use a small test_size for a quick demonstration
    test_dataset = OpenR1MathDataset(split="test", test_size=500)

    for i in range(0, 100):
        item = test_dataset[i]
        print(f"\nExample {i + 1}:")
        # print("Problem:", item["problem"])
        print("Answer:", item["answer"])

        print(parse(f"${item["answer"]}$"))
