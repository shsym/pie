from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset
from rewards import extract_answer, format_reward, accuracy_reward


INSTRUCTION = "Please reason step by step, and put your final answer within \boxed{}."


class OpenR1MathDataset(Dataset):
    """Prepares the OpenR1-Math-220k dataset for training and testing."""

    def __init__(self, split: str = "train", test_size: int = 100):
        # Load the dataset from Hugging Face
        dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

        # Shuffle the dataset for an unbiased split
        shuffled_dataset = dataset.shuffle(seed=42)

        # Use .select() for proper splitting
        if split == "train":
            self.data = shuffled_dataset.select(range(len(shuffled_dataset) - test_size))
        else:
            self.data = shuffled_dataset.select(range(len(shuffled_dataset) - test_size, len(shuffled_dataset)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "problem": item["problem"] + "\n" + INSTRUCTION,
            "answer": item["answer"],
            "verifier": lambda response: verify_answer(response, item["answer"]),
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
    test_dataset = OpenR1MathDataset(split="test", test_size=5)

    # Get the first sample from the test set
    sample_item = test_dataset[0]
    problem = sample_item["problem"]
    verifier = sample_item["verifier"]

    print("\n" + "=" * 50)
    print(f"Testing with Problem:\n{problem}")
    print(f"Expected Answer (from dataset): {sample_item['answer']}")
    print("=" * 50 + "\n")

    # Define a few example model responses to test
    test_responses = {
        "Perfect Response": "<think>\nThe user wants the answer. I will calculate it and put it in the answer tag.\n</think>\n<answer>\n\\boxed{20}\n</answer>",
        "Correct Answer, Bad Format": "Here is the result: \\boxed{20}",
        "Correct Format, Wrong Answer": "<think>\nI think the answer is 10.\n</think>\n<answer>\n\\boxed{10}\n</answer>",
        "Malformed (No Answer Tag)": "<think>\nI will solve this now.\n</think>\nI am pretty sure the answer is 20.",
        "Correct Answer, Unboxed": "<think>\nOkay, the answer is 20.\n</think>\n<answer>20</answer>",
    }

    # Manually override the gold answer for this specific test case for consistency
    # (The first item in the shuffled dataset is "What is $10+10$?")
    sample_item['answer'] = "\\boxed{20}"
    verifier = lambda response: verify_answer(response, sample_item['answer'])

    for name, response in test_responses.items():
        print(f"--- Testing: {name} ---")
        print(f"Response:\n{response}")
        reward_dict = verifier(response)
        print(f"Resulting Rewards: {reward_dict}\n")
