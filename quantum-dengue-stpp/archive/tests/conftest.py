"""
Pytest configuration and fixtures.
"""
import sys
import os
import pytest
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_events_df():
    """Sample events DataFrame for testing."""
    return pd.DataFrame({
        'lat': np.random.uniform(-6, 23, 100),
        'lon': np.random.uniform(95, 141, 100),
        'timestamp': pd.date_range('2020-01-01', periods=100, freq='ME'),
        'case_count': np.random.randint(0, 100, 100),
        'region': [f'region_{i}' for i in range(100)],
        'country': np.random.choice(['Vietnam', 'Thailand', 'Indonesia'], 100),
    })


@pytest.fixture
def sample_grid():
    """Sample spatial grid for testing."""
    return np.random.rand(16, 16, 50)


@pytest.fixture
def sample_tensor():
    """Sample tensor for model testing."""
    return torch.randn(4, 12, 20, 20)


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()
