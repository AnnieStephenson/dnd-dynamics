"""
D&D Campaign Metrics

Available metrics:
- basic: Campaign statistics (time intervals, post lengths, dice rolls)
- jaccard: Lexical cohesion via Jaccard similarity
- semantic: SBERT semantic distance, session novelty
- dsi: BERT-based Divergent Semantic Integration (GPU/MPS accelerated)
"""

from .basic import analyze_basic_metrics
from .jaccard import analyze_jaccard
from .semantic import analyze_semantic
from .dsi import analyze_dsi


def analyze_all(data, metrics=None, **kwargs):
    """
    Run multiple metrics on campaign data.

    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        metrics: List of metric names to run. If None, runs all metrics.
                 Available: 'basic', 'jaccard', 'semantic', 'dsi'
        **kwargs: Additional arguments passed to each metric function
                  (e.g., force_refresh=True, show_progress=False)

    Returns:
        Dict mapping metric name to results
    """
    available = {
        'basic': analyze_basic_metrics,
        'jaccard': analyze_jaccard,
        'semantic': analyze_semantic,
        'dsi': analyze_dsi,
    }

    if metrics is None:
        metrics = list(available.keys())

    return {name: available[name](data, **kwargs) for name in metrics if name in available}
