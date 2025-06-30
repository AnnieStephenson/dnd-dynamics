"""
Configuration file for pytest.

This file contains shared fixtures and configuration for all tests in the test suite.
"""

import pytest
import sys
import os

# Ensure the main dnd_analysis module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_environment():
    """
    Session-scoped fixture to verify test environment is set up correctly.
    """
    try:
        import dnd_analysis
        import pandas as pd
        import numpy as np
        return {
            "dnd_analysis_available": True,
            "pandas_available": True,
            "numpy_available": True
        }
    except ImportError as e:
        pytest.fail(f"Required dependencies not available: {e}")


def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )