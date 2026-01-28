"""
D&D Campaign Metrics

Available metrics:
- basic: Campaign statistics (time intervals, post lengths, dice rolls)
- jaccard: Lexical cohesion via Jaccard similarity
- semantic: SBERT semantic distance, session novelty
- dsi: BERT-based Divergent Semantic Integration (GPU/MPS accelerated)
- llm_judge_creativity: LLM-based creativity scoring (Novelty, Value, Adherence, Resonance)
- conflict: LLM-based interpersonal conflict detection and intensity rating
- humor: LLM-based humor/joke detection and originality rating
- cooperation: LLM-based cooperation detection and depth rating
- norms: LLM-based social norm detection (establishing, following, violating, enforcing)
- collab_creativity: LLM-based collaborative creativity detection and rating
"""

from .basic import analyze_basic_metrics
from .jaccard import analyze_jaccard
from .semantic import analyze_semantic
from .dsi import analyze_dsi
from .llm_judge_creativity import analyze_llm_judge_creativity
from .llm_conflict import analyze_conflict
from .llm_humor import analyze_humor
from .llm_cooperation import analyze_cooperation
from .llm_norms import analyze_norms
from .llm_collab_creativity import analyze_collab_creativity
from .result import MetricResult


def analyze_all(data, metrics=None, **kwargs):
    """
    Run multiple metrics on campaign data.

    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        metrics: List of metric names to run. If None, runs default metrics
                 (excludes LLM-based metrics due to API costs).
                 Available: 'basic', 'jaccard', 'semantic', 'dsi', 'llm_judge_creativity', 'conflict', 'humor', 'cooperation', 'norms', 'collab_creativity'
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
        'llm_judge_creativity': analyze_llm_judge_creativity,
        'conflict': analyze_conflict,
        'humor': analyze_humor,
        'cooperation': analyze_cooperation,
        'norms': analyze_norms,
        'collab_creativity': analyze_collab_creativity,
    }

    # Default excludes LLM-based metrics due to API costs
    default_metrics = ['basic', 'jaccard', 'semantic', 'dsi']

    if metrics is None:
        metrics = default_metrics

    return {name: available[name](data, **kwargs) for name in metrics if name in available}
