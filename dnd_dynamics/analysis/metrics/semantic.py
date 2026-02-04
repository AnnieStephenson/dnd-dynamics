"""
Semantic Cohesion Analysis for D&D Campaigns

This module provides sentence embedding-based semantic analysis functions for measuring
group cohesion in D&D gameplay using cosine similarity.

Key Metrics:
- Sequential Cohesion: Similarity between consecutive posts (higher = more cohesive flow)
- Session Cohesion: Within-session semantic similarity statistics

Supports SBERT and MPNet models via sentence-transformers. GPU/MPS accelerated.
For DSI (Divergent Semantic Integration), see the dsi module.
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
from sentence_transformers import SentenceTransformer

from dnd_dynamics import config
from . import _cache
from .result import MetricResult


# Global cache for embedding models to avoid reloading
_embedding_model_cache = {}


def _get_device():
    """Get best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _get_embedding_model(model_name=None):
    """Get or load embedding model with device placement."""
    if model_name is None:
        model_name = config.SENTENCE_EMBEDDING_MODEL

    if model_name not in _embedding_model_cache:
        device = _get_device()
        print(f"Loading embedding model {model_name} on {device}...")
        model = SentenceTransformer(model_name, device=str(device))
        _embedding_model_cache[model_name] = {'model': model, 'device': device}

    return _embedding_model_cache[model_name]


def clear_embedding_cache():
    """Clear the embedding model cache to free memory."""
    global _embedding_model_cache
    _embedding_model_cache.clear()
    print("Embedding model cache cleared")


# ===================================================================
# SENTENCE EMBEDDING COHESION ANALYSIS
# ===================================================================


def get_embeddings(df: pd.DataFrame,
                  model_name: str = None,
                  text_col: str = "text",
                  cache_dir: Optional[str] = None,
                  batch_size: int = None) -> np.ndarray:
    """
    Generate and cache sentence embeddings for text data. GPU/MPS accelerated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data
    model_name : str, default None
        Name of the sentence-transformers model to use. If None, uses config.SENTENCE_EMBEDDING_MODEL
    text_col : str, default "text"
        Column name containing text to embed
    cache_dir : Optional[str], default None
        Directory to cache embeddings. If None, uses data/processed/embeddings_cache/
    batch_size : int, default None
        Batch size for encoding. If None, uses config.SENTENCE_EMBEDDING_BATCH_SIZE

    Returns
    -------
    np.ndarray
        Array of embeddings with shape (n_texts, embedding_dim)
    """
    if model_name is None:
        model_name = config.SENTENCE_EMBEDDING_MODEL
    if batch_size is None:
        batch_size = config.SENTENCE_EMBEDDING_BATCH_SIZE

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

    # Get model with GPU/MPS support
    model_cache = _get_embedding_model(model_name)
    model = model_cache['model']

    # Handle empty or very short texts
    processed_texts = [text if len(text.strip()) > 0 else "[EMPTY]" for text in texts]

    # Generate embeddings in batches
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
                           model_name: str = None,
                           cache_dir: Optional[str] = None,
                           labels_to_process: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate and cache sentence embeddings for text data separated by label. GPU/MPS accelerated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data with label information
    model_name : str, default None
        Name of the sentence-transformers model to use. If None, uses config.SENTENCE_EMBEDDING_MODEL
    cache_dir : Optional[str], default None
        Directory to cache embeddings. If None, uses data/processed/embeddings_cache/
    labels_to_process : List[str], optional
        Which labels to process. If None, processes all available labels.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping label types to embedding arrays
    """
    if model_name is None:
        model_name = config.SENTENCE_EMBEDDING_MODEL

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


def sequential_cohesion(df: pd.DataFrame,
                        embeddings: Optional[np.ndarray] = None,
                        window: int = 1) -> pd.Series:
    """
    Calculate semantic cohesion between each post and the one `window` steps back.

    Higher values indicate more semantically similar/cohesive sequential posts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text data (should be sorted by time)
    embeddings : np.ndarray, optional
        Precomputed embeddings. If None, will generate using get_embeddings()
    window : int, default 1
        Number of steps back to compare (1 = previous post)

    Returns
    -------
    pd.Series
        Series of cosine similarities with same index as df (values in [-1, 1], typically [0, 1])
    """
    if embeddings is None:
        embeddings = get_embeddings(df)

    if len(embeddings) != len(df):
        raise ValueError(f"Embeddings length ({len(embeddings)}) doesn't match DataFrame length ({len(df)})")

    # Calculate pairwise similarities
    similarities = []

    for i in range(len(embeddings)):
        if i < window:
            # Not enough previous posts
            similarities.append(np.nan)
        else:
            # Calculate similarity to post `window` steps back
            current_emb = embeddings[i].reshape(1, -1)
            previous_emb = embeddings[i - window].reshape(1, -1)

            sim = cosine_similarity(current_emb, previous_emb)[0, 0]
            similarities.append(sim)

    return pd.Series(similarities, index=df.index, name=f'sequential_cohesion_w{window}')


def session_cohesion(df: pd.DataFrame,
                     embeddings: Optional[np.ndarray] = None,
                     session_col: str = "session_id") -> pd.DataFrame:
    """
    Calculate session-level cohesion based on pairwise semantic similarities.

    Higher values indicate more semantically cohesive sessions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with session identifiers
    embeddings : np.ndarray, optional
        Precomputed embeddings
    session_col : str, default "session_id"
        Column name containing session identifiers

    Returns
    -------
    pd.DataFrame
        Session-level statistics with columns: mean_similarity, median_similarity, etc.
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

        # Calculate all pairwise similarities within session
        similarities = cosine_similarity(session_embeddings)

        # Get upper triangle (excluding diagonal) to avoid duplicates
        upper_triangle = np.triu(similarities, k=1)
        # For similarities, we want all non-diagonal upper triangle values
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        pairwise_sims = similarities[mask]

        if len(pairwise_sims) > 0:
            stats = {
                'session_id': session_id,
                'mean_similarity': np.mean(pairwise_sims),
                'median_similarity': np.median(pairwise_sims),
                'min_similarity': np.min(pairwise_sims),
                'std_similarity': np.std(pairwise_sims),
                'post_count': len(session_embeddings),
                'total_comparisons': len(pairwise_sims)
            }
            session_stats.append(stats)

    return pd.DataFrame(session_stats)


def _calculate_player_sequential_cohesion(df: pd.DataFrame,
                                          embeddings: np.ndarray,
                                          player: str,
                                          window: int = 1) -> np.ndarray:
    """
    Calculate sequential cohesion for a single player's posts.

    Compares each of the player's posts to their previous post (ignoring
    posts from other players in between).

    Parameters
    ----------
    df : pd.DataFrame
        Full campaign DataFrame (used to get player mask)
    embeddings : np.ndarray
        Precomputed embeddings for all posts in df
    player : str
        Player name to analyze
    window : int, default 1
        Number of steps back to compare

    Returns
    -------
    np.ndarray
        Array of cosine similarities between consecutive player posts
    """
    player_mask = df['player'] == player
    player_indices = df.index[player_mask].tolist()

    if len(player_indices) < 2:
        return np.array([])

    # Get embeddings for this player's posts in order
    # Convert DataFrame index to positional index for embeddings array
    positional_indices = [df.index.get_loc(idx) for idx in player_indices]
    player_embeddings = embeddings[positional_indices]

    # Calculate similarity between consecutive posts
    scores = []
    for i in range(window, len(player_embeddings)):
        current_emb = player_embeddings[i].reshape(1, -1)
        previous_emb = player_embeddings[i - window].reshape(1, -1)
        sim = cosine_similarity(current_emb, previous_emb)[0, 0]
        scores.append(sim)

    return np.array(scores)


def _calculate_player_session_cohesion(df: pd.DataFrame,
                                       embeddings: np.ndarray,
                                       player: str,
                                       chunk_size: int = None) -> np.ndarray:
    """
    Calculate session-like cohesion for a single player's posts.

    Groups the player's messages into fixed-size chunks (pseudo-sessions)
    and calculates within-chunk pairwise similarity.

    Parameters
    ----------
    df : pd.DataFrame
        Full campaign DataFrame (used to get player mask)
    embeddings : np.ndarray
        Precomputed embeddings for all posts in df
    player : str
        Player name to analyze
    chunk_size : int, optional
        Number of messages per chunk. Defaults to config.MESSAGES_PER_SESSION

    Returns
    -------
    np.ndarray
        Array of mean pairwise similarities per chunk
    """
    if chunk_size is None:
        chunk_size = config.MESSAGES_PER_SESSION

    player_mask = df['player'] == player
    player_indices = df.index[player_mask].tolist()

    if len(player_indices) < 2:
        return np.array([])

    # Get embeddings for this player's posts in order
    positional_indices = [df.index.get_loc(idx) for idx in player_indices]
    player_embeddings = embeddings[positional_indices]

    # Group into chunks and calculate within-chunk similarity
    chunk_similarities = []
    for start in range(0, len(player_embeddings), chunk_size):
        chunk_emb = player_embeddings[start:start + chunk_size]

        if len(chunk_emb) < 2:
            continue

        # Calculate all pairwise similarities within chunk
        similarities = cosine_similarity(chunk_emb)

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        pairwise_sims = similarities[mask]

        if len(pairwise_sims) > 0:
            chunk_similarities.append(np.mean(pairwise_sims))

    return np.array(chunk_similarities)


def _analyze_single_campaign_semantic(df, campaign_id: str, show_progress: bool) -> Optional[MetricResult]:
    """
    Run semantic cohesion analysis for a single campaign.

    Args:
        df: Loaded campaign DataFrame
        campaign_id: Campaign identifier
        show_progress: Whether to show progress indicators

    Returns:
        MetricResult with semantic cohesion results, or None if insufficient data
    """
    # Check if campaign has sufficient data for analysis
    min_texts_required = 5
    if len(df) < min_texts_required:
        print(f"Skipping semantic analysis for {campaign_id}: insufficient data ({len(df)} texts, minimum {min_texts_required} required)")
        return None

    # Get embeddings for all text
    embeddings = get_embeddings(df)

    # Calculate sequential cohesion (post-to-post similarity)
    seq_cohesion = sequential_cohesion(df, embeddings=embeddings)

    # Analyze session cohesion (within-session similarity)
    sess_cohesion = session_cohesion(df, embeddings=embeddings)

    # Per-player metrics
    by_player = {}
    for player in df['player'].dropna().unique():
        player_seq = _calculate_player_sequential_cohesion(df, embeddings, player)
        player_sess = _calculate_player_session_cohesion(df, embeddings, player)
        by_player[player] = MetricResult(
            series={
                'sequential_cohesion': player_seq,
                'session_cohesion': player_sess,
            },
            summary={
                'mean_sequential_cohesion': float(np.mean(player_seq)) if len(player_seq) > 0 else np.nan,
                'mean_session_cohesion': float(np.mean(player_sess)) if len(player_sess) > 0 else np.nan,
            },
            by_player={},
            metadata={'post_count': int((df['player'] == player).sum())}
        )

    return MetricResult(
        series={
            'sequential_cohesion': seq_cohesion.dropna().values,
            'session_cohesion': sess_cohesion['mean_similarity'].values,
        },
        summary={
            'mean_sequential_cohesion': float(seq_cohesion.mean()),
            'mean_session_cohesion': float(sess_cohesion['mean_similarity'].mean()),
        },
        by_player=by_player,
        metadata={
            'campaign_id': campaign_id,
            'total_messages': len(df),
            'unique_players': df['player'].nunique(),
        }
    )


def analyze_semantic(data: Dict[str, pd.DataFrame],
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, MetricResult]:
    """
    Analyze semantic metrics for campaigns.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/semantic_results)
        force_refresh: Whether to force recomputation even if cached results exist

    Returns:
        Dict[campaign_id, MetricResult]
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")

    # Set default cache directory
    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'semantic_results')

    # Handle caching
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

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "semantic"
    )
