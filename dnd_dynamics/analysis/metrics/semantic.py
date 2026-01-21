"""
Semantic Analysis for D&D Campaigns

This module provides SBERT-based semantic analysis functions for measuring creativity,
novelty, and narrative evolution in D&D gameplay.

Key Metrics:
- Semantic Distance: Distance between consecutive posts using SBERT embeddings
- Session Novelty: Session-level semantic diversity statistics

For DSI (Divergent Semantic Integration), see the dsi module.
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from sentence_transformers import SentenceTransformer

from . import _cache


# ===================================================================
# SENTENCE EMBEDDING DISTANCE ANALYSIS (SBERT)
# ===================================================================


def get_embeddings(df: pd.DataFrame,
                  model_name: str = "all-MiniLM-L6-v2",
                  text_col: str = "text",
                  cache_dir: Optional[str] = None) -> np.ndarray:
    """
    Generate and cache Sentence-BERT embeddings for text data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data
    model_name : str, default "all-MiniLM-L6-v2"
        Name of the sentence-transformers model to use
    text_col : str, default "text"
        Column name containing text to embed
    cache_dir : Optional[str], default None
        Directory to cache embeddings. If None, uses data/processed/embeddings_cache/

    Returns
    -------
    np.ndarray
        Array of embeddings with shape (n_texts, embedding_dim)
    """
    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent  # Go up to repository root
        cache_dir = str(repo_root / 'data' / 'processed' / 'embeddings_cache')

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Create hash of texts and model name for cache key
    texts = df[text_col].fillna("").tolist()
    text_hash = hashlib.md5(str(texts + [model_name]).encode()).hexdigest()
    cache_file = cache_path / f"embeddings_{text_hash}.pkl"

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Generate embeddings
    print(f"Generating embeddings using model: {model_name}")
    model = SentenceTransformer(model_name)

    # Handle empty or very short texts
    processed_texts = [text if len(text.strip()) > 0 else "[EMPTY]" for text in texts]

    # Generate embeddings in batches for memory efficiency
    batch_size = 32
    embeddings = []

    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)

        if (i // batch_size) % 10 == 0:
            print(f"Processed {i + len(batch)}/{len(processed_texts)} texts")

    embeddings = np.vstack(embeddings)

    # Cache the results
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Cached embeddings to: {cache_file}")

    return embeddings


def get_embeddings_by_label(df: pd.DataFrame,
                           model_name: str = "all-MiniLM-L6-v2",
                           cache_dir: Optional[str] = None,
                           labels_to_process: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate and cache Sentence-BERT embeddings for text data separated by label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data with label information
    model_name : str, default "all-MiniLM-L6-v2"
        Name of the sentence-transformers model to use
    cache_dir : Optional[str], default None
        Directory to cache embeddings. If None, uses data/processed/embeddings_cache/
    labels_to_process : List[str], optional
        Which labels to process. If None, processes all available labels.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping label types to embedding arrays
    """
    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent  # Go up to repository root
        cache_dir = str(repo_root / 'data' / 'processed' / 'embeddings_cache')

    if labels_to_process is None:
        labels_to_process = ['in-character', 'out-of-character', 'mixed']

    # Map labels to DataFrame columns
    label_columns = {
        'in-character': 'in_character_text',
        'out-of-character': 'out_of_character_text',
        'mixed': 'mixed_text'
    }

    embeddings_by_label = {}

    for label in labels_to_process:
        if label in label_columns:
            text_col = label_columns[label]

            # Filter to only messages with content for this label
            label_df = df[df[text_col].str.len() > 0].copy()

            if len(label_df) > 0:
                print(f"Processing {len(label_df)} messages for label: {label}")
                embeddings = get_embeddings(label_df, model_name=model_name,
                                          text_col=text_col, cache_dir=cache_dir)
                embeddings_by_label[label] = embeddings
            else:
                print(f"No content found for label: {label}")
                embeddings_by_label[label] = np.array([])

    return embeddings_by_label


def semantic_distance(df: pd.DataFrame,
                     embeddings: Optional[np.ndarray] = None,
                     window: int = 1,
                     metric: str = "cosine",
                     normalize: bool = True) -> pd.Series:
    """
    Calculate semantic distance between each post and the one `window` steps back.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text data (should be sorted by time)
    embeddings : np.ndarray, optional
        Precomputed embeddings. If None, will generate using get_embeddings()
    window : int, default 1
        Number of steps back to compare (1 = previous post)
    metric : str, default "cosine"
        Distance metric: "cosine" or "euclidean"
    normalize : bool, default True
        Whether to normalize distances to [0,1] range

    Returns
    -------
    pd.Series
        Series of distances with same index as df
    """
    if embeddings is None:
        embeddings = get_embeddings(df)

    if len(embeddings) != len(df):
        raise ValueError(f"Embeddings length ({len(embeddings)}) doesn't match DataFrame length ({len(df)})")

    # Calculate pairwise distances
    distances = []

    for i in range(len(embeddings)):
        if i < window:
            # Not enough previous posts
            distances.append(np.nan)
        else:
            # Calculate distance to post `window` steps back
            current_emb = embeddings[i].reshape(1, -1)
            previous_emb = embeddings[i - window].reshape(1, -1)

            dist = pairwise_distances(current_emb, previous_emb, metric=metric)[0, 0]
            distances.append(dist)

    distances = np.array(distances)

    # Normalize to [0,1] if requested
    if normalize and not np.all(np.isnan(distances)):
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) > 0:
            scaler = MinMaxScaler()
            distances_scaled = scaler.fit_transform(valid_distances.reshape(-1, 1)).flatten()

            # Put scaled values back
            result = np.full_like(distances, np.nan)
            result[~np.isnan(distances)] = distances_scaled
            distances = result

    return pd.Series(distances, index=df.index, name=f'semantic_distance_w{window}')


def session_novelty(df: pd.DataFrame,
                   embeddings: Optional[np.ndarray] = None,
                   session_col: str = "session_id",
                   metric: str = "cosine") -> pd.DataFrame:
    """
    Calculate session-level novelty metrics based on pairwise semantic distances.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with session identifiers
    embeddings : np.ndarray, optional
        Precomputed embeddings
    session_col : str, default "session_id"
        Column name containing session identifiers
    metric : str, default "cosine"
        Distance metric for embeddings

    Returns
    -------
    pd.DataFrame
        Session-level statistics with columns: mean, median, max, std, count
    """
    if embeddings is None:
        embeddings = get_embeddings(df)

    session_stats = []

    for session_id in df[session_col].unique():
        if pd.isna(session_id):
            continue

        session_mask = df[session_col] == session_id
        session_embeddings = embeddings[session_mask]

        if len(session_embeddings) < 2:
            # Skip sessions with only one post
            continue

        # Calculate all pairwise distances within session
        distances = pairwise_distances(session_embeddings, metric=metric)

        # Get upper triangle (excluding diagonal) to avoid duplicates
        upper_triangle = np.triu(distances, k=1)
        pairwise_dists = upper_triangle[upper_triangle > 0]

        if len(pairwise_dists) > 0:
            stats = {
                'session_id': session_id,
                'mean_distance': np.mean(pairwise_dists),
                'median_distance': np.median(pairwise_dists),
                'max_distance': np.max(pairwise_dists),
                'std_distance': np.std(pairwise_dists),
                'post_count': len(session_embeddings),
                'total_comparisons': len(pairwise_dists)
            }
            session_stats.append(stats)

    return pd.DataFrame(session_stats)


def _analyze_single_campaign_semantic(df, campaign_id: str, show_progress: bool) -> Dict:
    """
    Run semantic analysis functions for a single campaign.

    Args:
        df: Loaded campaign DataFrame
        campaign_id: Campaign identifier
        show_progress: Whether to show progress indicators

    Returns:
        Dict with semantic analysis results
    """
    # Check if campaign has sufficient data for analysis
    min_texts_required = 5
    if len(df) < min_texts_required:
        print(f"Skipping semantic analysis for {campaign_id}: insufficient data ({len(df)} texts, minimum {min_texts_required} required)")
        return None

    campaign_results = {}

    # Get embeddings for all text
    embeddings = get_embeddings(df)
    campaign_results['embeddings'] = embeddings

    # Calculate semantic distances
    semantic_dists = semantic_distance(df, embeddings=embeddings)
    campaign_results['semantic_distances'] = semantic_dists

    # Analyze session novelty
    novelty_results = session_novelty(df, embeddings=embeddings)
    campaign_results['session_novelty'] = novelty_results

    # Store campaign metadata
    campaign_results['metadata'] = {
        'campaign_id': campaign_id,
        'total_messages': len(df),
        'unique_players': df['player'].nunique(),
        'date_range': {
            'start': df['date'].min().isoformat() if not df['date'].isna().all() else None,
            'end': df['date'].max().isoformat() if not df['date'].isna().all() else None
        }
    }

    return campaign_results


def analyze_semantic(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Union[Dict, Dict[str, Dict]]:
    """
    Analyze semantic metrics for single or multiple campaigns.

    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/semantic_results)
        force_refresh: Whether to force recomputation even if cached results exist

    Returns:
        Dict of results for single campaign, or Dict[campaign_id, results] for multiple
    """
    if isinstance(data, pd.DataFrame):
        # Single campaign analysis - no caching for single campaigns
        campaign_id = "not specified"
        return _analyze_single_campaign_semantic(data, campaign_id, show_progress=False)

    elif isinstance(data, dict):
        # Multiple campaign analysis with caching support

        # Set default cache directory
        if cache_dir is None:
            repo_root = Path(__file__).parent.parent.parent.parent  # Go up to repository root
            cache_dir = str(repo_root / 'data' / 'processed' / 'semantic_results')

        # Handle caching using helper function
        cached_results, data_to_process = _cache.handle_multi_campaign_caching(
            data, cache_dir, force_refresh, show_progress, "semantic"
        )

        # Process missing campaigns
        new_results = {}
        if data_to_process:
            if show_progress and len(data_to_process) > 1:
                iterator = tqdm(data_to_process.items(), desc="Analyzing semantic", total=len(data_to_process))
            else:
                iterator = data_to_process.items()

            for campaign_id, df in iterator:
                result = _analyze_single_campaign_semantic(
                    df, campaign_id, show_progress=False
                )
                # Only include campaigns that had sufficient data
                if result is not None:
                    new_results[campaign_id] = result

        # Save new results and combine with cached results
        return _cache.save_new_results_and_combine(
            cached_results, new_results, cache_dir, show_progress, "semantic"
        )

    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.DataFrame or Dict[str, pd.DataFrame]")
