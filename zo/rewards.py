import re
from typing import Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completion: str, solution: str) -> Optional[float]:
    """
    Reward function that checks if the completion is mathematically the same as the ground truth.
    It first attempts a strict parse (requiring a boxed answer) and falls back to a lenient parse.
    """

    if completion.lower().strip() == solution.lower().strip():
        return 1.0

    gold_parsed = parse(
        solution,
        #extraction_mode="first_match",
    )
    if not gold_parsed:
        # If the gold solution is not parseable, skip this example
        print("Failed to parse gold solution: ", solution)
        return None

    # 1. First, try a strict configuration that requires a boxed answer and perfect syntax.
    strict_config = LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,  # e.g., requires \sin(x), not \sin x
            basic_latex=True,
            #equations=True,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,  # Must be in \boxed{} or similar
    )
    answer_parsed = parse(
        completion, extraction_config=[strict_config], extraction_mode="first_match"
    )

    # 2. If strict parsing fails, fall back to a more lenient configuration.
    if not answer_parsed:
        lenient_config = LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=True,  # e.g., accepts \sin x
                basic_latex=True,
                #equations=True,
                boxed="all",
                units=True,
            ),
            try_extract_without_anchor=True,  # Can find answers without \boxed{}
        )
        answer_parsed = parse(
            completion,
            extraction_config=[lenient_config],
            extraction_mode="first_match",
        )

    # Compute binary rewards if verifiable, `None` otherwise
    try:
        reward = float(verify(gold_parsed, answer_parsed))
    except Exception as e:
        print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        reward = None

    return reward


def format_reward(completion: str) -> float:
    """
    Reward function that checks if the reasoning process is enclosed within <think>
    and </think> tags, while the final answer is enclosed within <answer> and </answer> tags.
    This check is flexible regarding whitespace.
    """
    # Flexible pattern allowing whitespace and searching anywhere in the stripped string
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    # Use re.search on the stripped string to find the pattern anywhere
    m = re.search(pattern, completion.strip(), re.DOTALL)
    return 1.0 if m else 0.0


def extract_answer(completion: str) -> Optional[str]:
    """Extracts the text between <answer> and </answer> tags."""
    # Use re.search with re.DOTALL to find content across multiple lines
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        # .group(1) gets the content of the first capturing group
        # .strip() removes leading/trailing whitespace, including newlines
        return match.group(1).strip()
    return None
