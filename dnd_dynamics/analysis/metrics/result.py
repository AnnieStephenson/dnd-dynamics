"""
MetricResult - Standardized container for metric analysis results.

All metric analysis functions return MetricResult instances to ensure
consistent access patterns across the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class MetricResult:
    """
    Standardized container for metric analysis results.

    Attributes:
        series: Arrays of values computed over sequential chunks of the campaign
                (per-message, per-session, per-scene depending on metric).
                Used for histogram plotting and time-series analysis.
        summary: Single values summarizing the whole campaign (often time-averaged).
        by_player: Optional player-level breakdowns (each is a MetricResult).
        metadata: Information about the analysis (campaign_id, counts, etc.).
    """
    series: Dict[str, np.ndarray] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)
    by_player: Dict[str, 'MetricResult'] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def list_series(self) -> list:
        """List available series metrics."""
        return list(self.series.keys())

    def list_summary(self) -> list:
        """List available summary metrics."""
        return list(self.summary.keys())
