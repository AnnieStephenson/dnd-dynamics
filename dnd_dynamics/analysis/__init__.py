"""
D&D Analysis Module

This module provides comprehensive analysis tools for D&D gameplay logs.
The module is organized into focused submodules for better maintainability:

- data_loading: Data loading and preprocessing functions
- data_correction: LLM-based data correction utilities
- metrics: Campaign metrics (basic stats, jaccard similarity, semantic analysis)
- plot_utils: Visualization functions

Usage:
    from dnd_dynamics.analysis import data_loading as dl
    from dnd_dynamics.analysis import metrics
    from dnd_dynamics.analysis import plot_utils as plots

    # Run all metrics
    results = metrics.analyze_all(data)

    # Or run specific metrics
    from dnd_dynamics.analysis.metrics import basic, jaccard, semantic
"""

from . import metrics
from .metrics import analyze_all

__version__ = "0.0.0"
