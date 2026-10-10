"""Shared fixtures. The trained forest is session-scoped: fitting it costs about
twelve seconds on two cores and several test modules need it, so fitting it once
is the difference between a two-minute suite and a six-minute one."""

from __future__ import annotations

import pytest

from telemdrift.learned import build_training_set, train_learned_detector


@pytest.fixture(scope="session")
def small_training_set():
    """A deliberately small labelled set: four seeds, enough to fit and score."""
    return build_training_set(seeds=range(58_001, 58_005))


@pytest.fixture(scope="session")
def small_model(small_training_set):
    """A small forest, 40 trees. Not the one the validation scripts publish."""
    return train_learned_detector(small_training_set, n_estimators=40, max_depth=6)
