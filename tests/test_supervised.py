import pytest

from countdown_grpo.prepare_supervised import supervised_pairs
from countdown_grpo.verifier import verify_completion


def test_supervised_branch_rejects_dev_and_verifies_train_targets():
    task = {"split": "dev", "nums": [2, 3, 4], "target": 9,
            "task_id": "example", "prompt": "Numbers: 2, 3, 4. Target: 9"}
    with pytest.raises(ValueError, match="training-only"):
        supervised_pairs([task], 1, 42)
    task["split"] = "train"
    pair = supervised_pairs([task], 1, 42)[0]
    assert pair["prompt"] == task["prompt"]
    assert verify_completion(pair["completion"], 9, [2, 3, 4]).valid
