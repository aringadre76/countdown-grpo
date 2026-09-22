import pytest

from countdown_grpo.prepare_confirmation import frozen_confirmation_size


def test_frozen_confirmation_size_accepts_new_separate_suite_fields():
    assert frozen_confirmation_size(
        {
            "status": "frozen",
            "confirmation": {"source_tasks": 256, "fresh_tasks": 256},
        }
    ) == 256


def test_frozen_confirmation_size_accepts_legacy_field():
    assert frozen_confirmation_size(
        {"status": "frozen", "confirmation_tasks_per_suite": 128}
    ) == 128


@pytest.mark.parametrize(
    "design",
    [
        {
            "status": "draft",
            "confirmation": {"source_tasks": 256, "fresh_tasks": 256},
        },
        {
            "status": "frozen",
            "confirmation": {"source_tasks": 256, "fresh_tasks": 128},
        },
    ],
)
def test_frozen_confirmation_size_rejects_unfrozen_or_unequal_suites(design):
    with pytest.raises(ValueError):
        frozen_confirmation_size(design)
