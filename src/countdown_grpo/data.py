"""Dataset loading and prompt construction."""

from __future__ import annotations

from datasets import Dataset, load_dataset


DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"


def format_prompt(example: dict[str, object]) -> str:
    nums = ", ".join(str(number) for number in example["nums"])
    return (
        "Use each of the supplied numbers exactly once with +, -, *, or /. "
        "Parentheses are allowed. Return one legal expression inside "
        "<answer>...</answer>.\n"
        f"Numbers: {nums}\n"
        f"Target: {example['target']}\n"
        "Solution:"
    )


def load_countdown(seed: int = 42, test_size: int = 1024) -> tuple[Dataset, Dataset]:
    """Load the source dataset and make a deterministic train/test split."""

    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.filter(lambda row: len(row["nums"]) in (3, 4))
    split = dataset.train_test_split(test_size=test_size, seed=seed)

    def add_prompt(row: dict[str, object]) -> dict[str, str]:
        return {"prompt": format_prompt(row)}

    return split["train"].map(add_prompt), split["test"].map(add_prompt)
