"""
Caching utilities for multi-campaign metric analysis.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd


def _model_to_slug(model: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return model.replace("/", "_").replace(".", "-")


def _get_cache_filename(campaign_id: str, model: Optional[str] = None) -> str:
    """Get cache filename, optionally including model slug."""
    if model:
        return f"{campaign_id}__{_model_to_slug(model)}.pkl"
    return f"{campaign_id}.pkl"


def load_cached_results(cache_dir: str, campaign_ids: List[str], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Load cached results for specified campaigns.

    Args:
        cache_dir: Directory containing cache files
        campaign_ids: List of campaign IDs to load
        model: Optional model name. If provided, loads model-specific cache files.

    Returns:
        Dict mapping campaign_id to cached results (only for campaigns that have cache files)
    """
    cache_path = Path(cache_dir)
    cached_results = {}

    if not cache_path.exists():
        return cached_results

    for campaign_id in campaign_ids:
        cache_file = cache_path / _get_cache_filename(campaign_id, model)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_results[campaign_id] = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load cache for {campaign_id}: {e}")

    return cached_results


def save_cached_results(cache_dir: str, results: Dict[str, Any], model: Optional[str] = None) -> None:
    """
    Save results to individual campaign cache files.

    Args:
        cache_dir: Directory to store cache files
        results: Dict mapping campaign_id to results
        model: Optional model name. If provided, includes model in cache filename.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for campaign_id, result in results.items():
        cache_file = cache_path / _get_cache_filename(campaign_id, model)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save cache for {campaign_id}: {e}")


def get_missing_campaigns(cache_dir: str, campaign_ids: List[str], model: Optional[str] = None) -> List[str]:
    """
    Return campaign IDs that don't have cached results.

    Args:
        cache_dir: Directory containing cache files
        campaign_ids: List of campaign IDs to check
        model: Optional model name. If provided, checks for model-specific cache files.

    Returns:
        List of campaign IDs that need computation
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return campaign_ids

    missing = []
    for campaign_id in campaign_ids:
        cache_file = cache_path / _get_cache_filename(campaign_id, model)
        if not cache_file.exists():
            missing.append(campaign_id)

    return missing


def get_cache_status(cache_dir: str, campaign_ids: List[str], model: Optional[str] = None) -> Dict[str, bool]:
    """
    Get cache status for a list of campaigns.

    Args:
        cache_dir: Directory containing cache files
        campaign_ids: List of campaign IDs to check
        model: Optional model name. If provided, checks for model-specific cache files.

    Returns:
        Dict mapping campaign_id to cache status (True if cached, False if not)
    """
    cache_path = Path(cache_dir)
    status = {}

    for campaign_id in campaign_ids:
        if cache_path.exists():
            cache_file = cache_path / _get_cache_filename(campaign_id, model)
            status[campaign_id] = cache_file.exists()
        else:
            status[campaign_id] = False

    return status


def handle_multi_campaign_caching(data: Dict[str, pd.DataFrame],
    cache_dir: str,
    force_refresh: bool,
    show_progress: bool,
    analysis_name: str,
    model: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Handle caching logic for multi-campaign analysis functions.

    This function manages the loading of cached results and determines which campaigns
    need to be processed. It's designed to be called from within analysis functions
    to handle caching transparently.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        cache_dir: Directory for caching results
        force_refresh: Whether to force recomputation even if cached results exist
        show_progress: Whether to show progress indicators
        analysis_name: Name of analysis for progress messages
        model: Optional model name. If provided, cache is model-specific.

    Returns:
        Tuple of (cached_results, data_to_process) where:
        - cached_results: Dict of previously cached results {campaign_id: results}
        - data_to_process: Dict of DataFrames that need analysis {campaign_id: df}
    """
    campaign_ids = list(data.keys())

    # Handle caching
    if not force_refresh:
        # Load cached results
        cached_results = load_cached_results(cache_dir, campaign_ids, model=model)
        missing_campaigns = get_missing_campaigns(cache_dir, campaign_ids, model=model)

        if show_progress and cached_results:
            print(f"📁 Loaded {len(cached_results)} cached {analysis_name} results")

        # Filter data to only missing campaigns
        data_to_process = {cid: data[cid] for cid in missing_campaigns}
    else:
        cached_results = {}
        data_to_process = data

    return cached_results, data_to_process


def save_new_results_and_combine(cached_results: Dict[str, Any], new_results: Dict[str, Any],
    cache_dir: str,
    show_progress: bool,
    analysis_name: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save new analysis results to cache and combine with cached results.

    Args:
        cached_results: Previously cached results
        new_results: Newly computed results
        cache_dir: Directory for caching results
        show_progress: Whether to show progress indicators
        analysis_name: Name of analysis for progress messages
        model: Optional model name. If provided, includes model in cache filename.

    Returns:
        Combined results dictionary
    """
    # Save new results to cache
    if new_results:
        save_cached_results(cache_dir, new_results, model=model)
        if show_progress:
            print(f"💾 Saved {len(new_results)} {analysis_name} results to cache")

    # Combine cached and new results
    results = {**cached_results, **new_results}
    return results
