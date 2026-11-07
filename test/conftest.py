"""
Pytest configuration and shared fixtures for surgical_scene_segmentation test suite.
"""
from __future__ import annotations

import os
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def configure_seed():
    """Ensure deterministic behaviour during testing."""
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def device():
    """Yields cuda device if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def require_cuda():
    """Skip test if CUDA is unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for this test.")

