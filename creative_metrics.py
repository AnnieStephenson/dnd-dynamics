"""
Creative Metrics Analysis for D&D Gameplay Logs

This module provides semantic analysis functions for measuring creativity, novelty,
and narrative evolution in D&D gameplay using embeddings and topic modeling.

Author: Claude Code Assistant
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import warnings

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    from gensim import corpora, models
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

def get_embeddings(df: pd.DataFrame, 
                  model_name: str = "all-MiniLM-L6-v2",
                  text_col: str = "text",
                  cache_dir: str = "embeddings_cache") -> np.ndarray:
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
    cache_dir : str, default "embeddings_cache"
        Directory to cache embeddings
        
    Returns
    -------
    np.ndarray
        Array of embeddings with shape (n_texts, embedding_dim)
        
    Raises
    ------
    ImportError
        If sentence-transformers is not installed
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
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
                           cache_dir: str = "embeddings_cache",
                           labels_to_process: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate and cache Sentence-BERT embeddings for text data separated by label.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data with label information
    model_name : str, default "all-MiniLM-L6-v2"
        Name of the sentence-transformers model to use
    cache_dir : str, default "embeddings_cache"
        Directory to cache embeddings
    labels_to_process : List[str], optional
        Which labels to process. If None, processes all available labels.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping label types to embedding arrays
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
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
    
    # If no session column, create one based on date gaps
    if session_col not in df.columns:
        print(f"No {session_col} column found. Creating sessions based on date gaps...")
        df = df.copy()
        df['session_id'] = _create_sessions_from_dates(df)
        session_col = 'session_id'
    
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

def topic_model(df: pd.DataFrame,
               n_topics: int = 20,
               engine: str = "bertopic",
               text_col: str = "text",
               embeddings: Optional[np.ndarray] = None,
               random_state: int = 42) -> Tuple[pd.Series, object]:
    """
    Assign topics to posts using BERTopic or LDA.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data
    n_topics : int, default 20
        Number of topics to extract
    engine : str, default "bertopic"
        Topic modeling engine: "bertopic" or "lda"
    text_col : str, default "text"
        Column containing text to model
    embeddings : np.ndarray, optional
        Precomputed embeddings (only used for BERTopic)
    random_state : int, default 42
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.Series, object]
        Series of topic assignments and fitted model object
    """
    texts = df[text_col].fillna("").tolist()
    
    if engine.lower() == "bertopic":
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available, falling back to LDA")
            return _topic_model_lda(texts, n_topics, random_state)
        return _topic_model_bertopic(texts, n_topics, embeddings, random_state)
    
    elif engine.lower() == "lda":
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for LDA. Install with: pip install gensim")
        return _topic_model_lda(texts, n_topics, random_state)
    
    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'bertopic' or 'lda'")

def _topic_model_bertopic(texts: List[str], 
                         n_topics: int,
                         embeddings: Optional[np.ndarray],
                         random_state: int) -> Tuple[pd.Series, 'BERTopic']:
    """BERTopic implementation."""
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    
    # Configure UMAP and HDBSCAN for reproducibility
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, 
                     metric='cosine', random_state=random_state)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', 
                           cluster_selection_method='eom')
    
    # Create BERTopic model
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        calculate_probabilities=False,
        verbose=True
    )
    
    # Fit model
    if embeddings is not None:
        topics, _ = topic_model.fit_transform(texts, embeddings)
    else:
        topics, _ = topic_model.fit_transform(texts)
    
    return pd.Series(topics, name='topic'), topic_model

def _topic_model_lda(texts: List[str], 
                    n_topics: int,
                    random_state: int) -> Tuple[pd.Series, models.LdaModel]:
    """LDA implementation using Gensim."""
    # Preprocess texts
    processed_texts = [simple_preprocess(text, deacc=True) for text in texts]
    
    # Remove empty documents
    processed_texts = [doc for doc in processed_texts if len(doc) > 0]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(doc) for doc in processed_texts]
    
    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=random_state,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # Get topic assignments
    topics = []
    for doc in corpus:
        topic_probs = lda_model.get_document_topics(doc)
        if topic_probs:
            # Assign to most probable topic
            best_topic = max(topic_probs, key=lambda x: x[1])[0]
            topics.append(best_topic)
        else:
            topics.append(-1)  # No topic assigned
    
    # Handle case where we had to remove empty documents
    if len(topics) < len(texts):
        # Fill in missing topics for empty documents
        full_topics = []
        topic_idx = 0
        for text in texts:
            if len(simple_preprocess(text, deacc=True)) > 0:
                full_topics.append(topics[topic_idx])
                topic_idx += 1
            else:
                full_topics.append(-1)
        topics = full_topics
    
    return pd.Series(topics, name='topic'), lda_model

def topic_transition_matrix(df: pd.DataFrame,
                           topic_col: str = "topic",
                           normalize: bool = True) -> pd.DataFrame:
    """
    Create Markov transition matrix P(next_topic | current_topic).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with topic assignments (should be sorted by time)
    topic_col : str, default "topic"
        Column containing topic assignments
    normalize : bool, default True
        Whether to normalize rows to sum to 1 (probabilities)
        
    Returns
    -------
    pd.DataFrame
        Transition matrix with topics as rows and columns
    """
    if topic_col not in df.columns:
        raise ValueError(f"Column '{topic_col}' not found in DataFrame")
    
    topics = df[topic_col].dropna()
    unique_topics = sorted(topics.unique())
    
    # Initialize transition matrix
    n_topics = len(unique_topics)
    transition_matrix = np.zeros((n_topics, n_topics))
    
    # Count transitions
    for i in range(len(topics) - 1):
        current_topic = topics.iloc[i]
        next_topic = topics.iloc[i + 1]
        
        if pd.notna(current_topic) and pd.notna(next_topic):
            current_idx = unique_topics.index(current_topic)
            next_idx = unique_topics.index(next_topic)
            transition_matrix[current_idx, next_idx] += 1
    
    # Convert to DataFrame
    transition_df = pd.DataFrame(
        transition_matrix,
        index=unique_topics,
        columns=unique_topics
    )
    
    # Normalize rows to probabilities
    if normalize:
        row_sums = transition_df.sum(axis=1)
        transition_df = transition_df.div(row_sums, axis=0).fillna(0)
    
    return transition_df

def topic_change_rate(df: pd.DataFrame,
                     topic_col: str = "topic",
                     window: int = 50) -> pd.Series:
    """
    Calculate fraction of topic switches within a rolling window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with topic assignments (should be sorted by time)
    topic_col : str, default "topic"
        Column containing topic assignments
    window : int, default 50
        Size of rolling window
        
    Returns
    -------
    pd.Series
        Rolling topic change rate
    """
    if topic_col not in df.columns:
        raise ValueError(f"Column '{topic_col}' not found in DataFrame")
    
    topics = df[topic_col]
    
    # Calculate topic changes (1 if topic changes, 0 if same)
    topic_changes = (topics != topics.shift(1)).astype(int)
    
    # Calculate rolling mean of changes
    change_rate = topic_changes.rolling(window=window, min_periods=1).mean()
    
    return change_rate

def _create_sessions_from_dates(df: pd.DataFrame, 
                               date_col: str = "date",
                               gap_hours: float = 6.0) -> pd.Series:
    """
    Create session IDs based on time gaps between posts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str, default "date"
        Column containing timestamps
    gap_hours : float, default 6.0
        Hours of gap to define new session
        
    Returns
    -------
    pd.Series
        Session IDs
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    dates = pd.to_datetime(df[date_col])
    time_diffs = dates.diff()
    
    # Mark new sessions where gap > threshold
    session_breaks = time_diffs > pd.Timedelta(hours=gap_hours)
    session_ids = session_breaks.cumsum()
    
    return session_ids

# Helper plotting functions
def plot_distance_timeline(df: pd.DataFrame, 
                          distance_col: str = "semantic_distance_w1",
                          date_col: str = "date",
                          rolling_window: int = 10) -> None:
    """
    Plot semantic distance over time with rolling average.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with distance and date columns
    distance_col : str
        Column containing semantic distances
    date_col : str
        Column containing timestamps
    rolling_window : int
        Window size for rolling average
    """
    import matplotlib.pyplot as plt
    
    # Calculate rolling average
    rolling_dist = df[distance_col].rolling(window=rolling_window, center=True).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[distance_col], alpha=0.3, label='Raw distance')
    plt.plot(df[date_col], rolling_dist, linewidth=2, label=f'{rolling_window}-post rolling average')
    plt.xlabel('Date')
    plt.ylabel('Semantic Distance')
    plt.title('Semantic Distance Timeline')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_topic_timeline(df: pd.DataFrame,
                       topic_col: str = "topic",
                       date_col: str = "date") -> None:
    """
    Plot topic evolution over time as colored timeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with topic and date columns
    topic_col : str
        Column containing topic assignments
    date_col : str
        Column containing timestamps
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create daily topic counts
    df_plot = df.copy()
    df_plot['date_day'] = pd.to_datetime(df_plot[date_col]).dt.date
    
    topic_counts = df_plot.groupby(['date_day', topic_col]).size().unstack(fill_value=0)
    
    # Plot stacked bar chart
    plt.figure(figsize=(15, 8))
    topic_counts.plot(kind='bar', stacked=True, 
                     colormap='tab20', figsize=(15, 8))
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.title('Topic Evolution Over Time')
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_session_novelty(session_stats: pd.DataFrame) -> None:
    """
    Plot session novelty statistics.
    
    Parameters
    ----------
    session_stats : pd.DataFrame
        Output from session_novelty() function
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mean distance distribution
    axes[0, 0].hist(session_stats['mean_distance'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Mean Semantic Distance')
    axes[0, 0].set_xlabel('Distance')
    
    # Max distance distribution
    axes[0, 1].hist(session_stats['max_distance'], bins=20, alpha=0.7)
    axes[0, 1].set_title('Max Semantic Distance')
    axes[0, 1].set_xlabel('Distance')
    
    # Post count vs mean distance
    axes[1, 0].scatter(session_stats['post_count'], session_stats['mean_distance'], alpha=0.6)
    axes[1, 0].set_xlabel('Posts per Session')
    axes[1, 0].set_ylabel('Mean Distance')
    axes[1, 0].set_title('Session Size vs Novelty')
    
    # Standard deviation
    axes[1, 1].hist(session_stats['std_distance'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Distance Std Deviation')
    axes[1, 1].set_xlabel('Std Distance')
    
    plt.tight_layout()
    plt.show()

def plot_topic_transitions(transition_matrix: pd.DataFrame,
                          top_n: int = 10) -> None:
    """
    Plot topic transition matrix as heatmap.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Output from topic_transition_matrix()
    top_n : int
        Number of top topics to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Select top N most common topics
    topic_counts = transition_matrix.sum(axis=1).sort_values(ascending=False)
    top_topics = topic_counts.head(top_n).index
    
    # Subset matrix
    subset_matrix = transition_matrix.loc[top_topics, top_topics]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(subset_matrix, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f'Topic Transition Matrix (Top {top_n} Topics)')
    plt.xlabel('Next Topic')
    plt.ylabel('Current Topic')
    plt.tight_layout()
    plt.show()


# Multi-Campaign Analysis Functions

def save_creativity_results(creativity_results: Dict, num_campaigns: int, cache_dir: str = 'campaign_stats_cache') -> str:
    """
    Save creativity analysis results to cache.
    
    Parameters
    ----------
    creativity_results : Dict
        Results from analyze_creativity_all_campaigns()
    num_campaigns : int
        Number of campaigns analyzed
    cache_dir : str
        Directory to store cached results
        
    Returns
    -------
    str
        Path to saved cache file
    """
    import os
    import pickle
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'creativity_analysis_{num_campaigns}_campaigns.pkl')
    
    with open(cache_file, 'wb') as f:
        pickle.dump(creativity_results, f)
    
    return cache_file


def load_creativity_results(num_campaigns: int, cache_dir: str = 'campaign_stats_cache') -> Optional[Dict]:
    """
    Load creativity analysis results from cache.
    
    Parameters
    ----------
    num_campaigns : int
        Number of campaigns to load
    cache_dir : str
        Directory containing cached results
        
    Returns
    -------
    Optional[Dict]
        Cached results or None if not found
    """
    import os
    import pickle
    
    cache_file = os.path.join(cache_dir, f'creativity_analysis_{num_campaigns}_campaigns.pkl')
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    return None


def load_or_compute_creativity_incremental(max_campaigns: int,
                                         data_file_path: str = 'Game-Data/data-labels.json',
                                         cache_dir: str = 'campaign_stats_cache',
                                         force_refresh: bool = False,
                                         show_progress: bool = True) -> Dict:
    """
    Smart loading for creativity metrics with incremental computation.
    
    This function implements intelligent caching by:
    1. Looking for existing creativity cache files
    2. Finding the largest cached result <= max_campaigns
    3. Computing only the additional campaigns needed
    4. Merging results efficiently
    
    Parameters
    ----------
    max_campaigns : int
        Maximum number of campaigns to analyze
    data_file_path : str
        Path to the JSON file containing campaign data
    cache_dir : str
        Directory to store cached results
    force_refresh : bool
        Whether to force recalculation even if cache exists
    show_progress : bool
        Whether to show progress bars
        
    Returns
    -------
    Dict
        Results from all creative metrics functions for each campaign
    """
    import json
    import os
    import glob
    
    if force_refresh:
        return analyze_creativity_all_campaigns(
            data_file_path=data_file_path,
            max_campaigns=max_campaigns,
            cache_dir=cache_dir,
            force_refresh=True,
            show_progress=show_progress
        )
    
    # Check for exact match first
    exact_cache = load_creativity_results(max_campaigns, cache_dir)
    if exact_cache is not None:
        if show_progress:
            print(f"Found exact cache for {max_campaigns} campaigns")
        return exact_cache
    
    # Look for existing creativity cache files
    cache_pattern = os.path.join(cache_dir, 'creativity_analysis_*_campaigns.pkl')
    cache_files = glob.glob(cache_pattern)
    
    if not cache_files:
        # No cache files found, compute from scratch
        if show_progress:
            print("No creativity cache files found, computing from scratch")
        return analyze_creativity_all_campaigns(
            data_file_path=data_file_path,
            max_campaigns=max_campaigns,
            cache_dir=cache_dir,
            force_refresh=False,
            show_progress=show_progress
        )
    
    # Find the largest cache file that's <= max_campaigns
    cached_counts = []
    for cache_file in cache_files:
        try:
            # Extract number from filename like: creativity_analysis_50_campaigns.pkl
            filename = os.path.basename(cache_file)
            num_str = filename.split('_')[2]  # Get the number part
            num_campaigns = int(num_str)
            if num_campaigns <= max_campaigns:
                cached_counts.append((num_campaigns, cache_file))
        except (IndexError, ValueError):
            continue
    
    if not cached_counts:
        # No suitable cache found, compute from scratch
        if show_progress:
            print("No suitable creativity cache found, computing from scratch")
        return analyze_creativity_all_campaigns(
            data_file_path=data_file_path,
            max_campaigns=max_campaigns,
            cache_dir=cache_dir,
            force_refresh=False,
            show_progress=show_progress
        )
    
    # Get the largest suitable cache
    cached_counts.sort(reverse=True)
    best_cached_count, best_cache_file = cached_counts[0]
    
    if show_progress:
        print(f"Found creativity cache with {best_cached_count} campaigns, need {max_campaigns}")
    
    if best_cached_count >= max_campaigns:
        # Cache has more than we need, load and subset
        cached_results = load_creativity_results(best_cached_count, cache_dir)
        if cached_results is not None:
            # Load data to get campaign order
            with open(data_file_path, 'r') as f:
                all_data = json.load(f)
            campaigns = list(all_data.keys())[:max_campaigns]
            
            # Subset cached results
            subset_results = {cid: cached_results[cid] for cid in campaigns if cid in cached_results}
            
            if len(subset_results) == max_campaigns:
                if show_progress:
                    print(f"Using subset of cached creativity results: {max_campaigns} campaigns")
                return subset_results
    
    # Need to compute additional campaigns
    cached_results = load_creativity_results(best_cached_count, cache_dir)
    if cached_results is None:
        # Cache load failed, compute from scratch
        if show_progress:
            print("Failed to load creativity cache, computing from scratch")
        return analyze_creativity_all_campaigns(
            data_file_path=data_file_path,
            max_campaigns=max_campaigns,
            cache_dir=cache_dir,
            force_refresh=False,
            show_progress=show_progress
        )
    
    # Load full data to get campaign ordering
    with open(data_file_path, 'r') as f:
        all_data = json.load(f)
    
    all_campaigns = list(all_data.keys())
    campaigns_needed = all_campaigns[:max_campaigns]
    campaigns_cached = list(cached_results.keys())
    campaigns_to_compute = [cid for cid in campaigns_needed if cid not in campaigns_cached]
    
    if not campaigns_to_compute:
        # We already have all needed campaigns
        result = {cid: cached_results[cid] for cid in campaigns_needed if cid in cached_results}
        if show_progress:
            print(f"All {max_campaigns} campaigns found in creativity cache")
        return result
    
    if show_progress:
        print(f"Computing creativity metrics for {len(campaigns_to_compute)} additional campaigns")
    
    # Compute additional campaigns
    additional_data = {cid: all_data[cid] for cid in campaigns_to_compute}
    additional_results = {}
    
    from tqdm import tqdm
    from dnd_analysis import load_dnd_data
    
    campaign_iterator = tqdm(campaigns_to_compute, desc="Computing additional creativity metrics") if show_progress else campaigns_to_compute
    
    for campaign_id in campaign_iterator:
        try:
            # Load single campaign data
            campaign_data = {campaign_id: all_data[campaign_id]}
            df = load_dnd_data(campaign_data)
            
            if len(df) == 0:
                continue
            
            campaign_results = {}
            
            # Get embeddings for all text (combined)
            embeddings = get_embeddings(df)
            campaign_results['embeddings'] = embeddings
            
            # Get label-aware embeddings
            try:
                label_embeddings = get_embeddings_by_label(df)
                campaign_results['label_embeddings'] = label_embeddings
            except Exception as e:
                if show_progress:
                    print(f"Label-aware embeddings failed for campaign {campaign_id}: {e}")
                campaign_results['label_embeddings'] = {}
            
            # Calculate semantic distances (combined)
            semantic_distances = semantic_distance(df, embeddings=embeddings)
            campaign_results['semantic_distances'] = semantic_distances
            
            # Calculate label-aware semantic distances
            label_semantic_distances = {}
            for label, label_emb in campaign_results['label_embeddings'].items():
                if len(label_emb) > 0:
                    # Create DataFrame for this label
                    label_col = {'in-character': 'in_character_text', 
                               'out-of-character': 'out_of_character_text', 
                               'mixed': 'mixed_text'}[label]
                    label_df = df[df[label_col].str.len() > 0].copy()
                    
                    if len(label_df) > 1:  # Need at least 2 messages for distance calculation
                        try:
                            label_distances = semantic_distance(label_df, embeddings=label_emb)
                            label_semantic_distances[label] = label_distances
                        except Exception as e:
                            if show_progress:
                                print(f"Label distance calculation failed for {label}: {e}")
            
            campaign_results['label_semantic_distances'] = label_semantic_distances
            
            # Analyze session novelty (combined)
            novelty_results = session_novelty(df, embeddings=embeddings)
            campaign_results['session_novelty'] = novelty_results
            
            # Topic modeling
            try:
                topics_series, topic_model_obj = topic_model(df, embeddings=embeddings)
                campaign_results['topic_model'] = {
                    'topics': topics_series,
                    'model': topic_model_obj
                }
                
                # Add topics to DataFrame for transition analysis
                df_with_topics = df.copy()
                df_with_topics['topic'] = topics_series
                
                # Topic transitions
                topic_transitions = topic_transition_matrix(df_with_topics, topic_col='topic')
                campaign_results['topic_transitions'] = topic_transitions
                
                # Topic change rate
                change_rate = topic_change_rate(df_with_topics, topic_col='topic')
                campaign_results['topic_change_rate'] = {'overall_rate': change_rate.mean(), 'series': change_rate}
                
            except Exception as e:
                if show_progress:
                    print(f"Topic modeling failed for campaign {campaign_id}: {e}")
                campaign_results['topic_model'] = None
                campaign_results['topic_transitions'] = None
                campaign_results['topic_change_rate'] = None
            
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
            
            additional_results[campaign_id] = campaign_results
            
        except Exception as e:
            if show_progress:
                print(f"Error processing campaign {campaign_id}: {e}")
            continue
    
    # Merge cached and new results
    merged_results = cached_results.copy()
    merged_results.update(additional_results)
    
    # Keep only the campaigns we need, in the correct order
    final_results = {cid: merged_results[cid] for cid in campaigns_needed if cid in merged_results}
    
    # Cache the new complete result
    cache_file = save_creativity_results(final_results, max_campaigns, cache_dir)
    if show_progress:
        print(f"Saved incremental creativity results to {cache_file}")
    
    return final_results


def analyze_creativity_all_campaigns(data_file_path: str = 'Game-Data/data-labels.json',
                                   max_campaigns: Optional[int] = None,
                                   cache_dir: str = 'creativity_cache',
                                   force_refresh: bool = False,
                                   show_progress: bool = True) -> Dict:
    """
    Apply all existing creative metric functions across multiple campaigns.
    
    Parameters
    ----------
    data_file_path : str
        Path to the JSON file containing campaign data
    max_campaigns : int, optional
        Maximum number of campaigns to analyze (None for all)
    cache_dir : str
        Directory to store cached results
    force_refresh : bool
        Whether to force recalculation even if cache exists
    show_progress : bool
        Whether to show progress bars
        
    Returns
    -------
    Dict
        Results from all creative metrics functions for each campaign
    """
    import json
    import os
    import pickle
    import hashlib
    from tqdm import tqdm
    from dnd_analysis import load_dnd_data
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load and validate data
    with open(data_file_path, 'r') as f:
        all_data = json.load(f)
    
    campaigns = list(all_data.keys())
    if max_campaigns:
        campaigns = campaigns[:max_campaigns]
    
    # Create cache file name
    cache_file = os.path.join(cache_dir, f'creativity_analysis_{len(campaigns)}_campaigns.pkl')
    
    # Check for existing cache
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)
            if show_progress:
                print(f"Loaded cached results for {len(campaigns)} campaigns from {cache_file}")
            return cached_results
        except Exception as e:
            if show_progress:
                print(f"Failed to load cache: {e}. Recomputing...")
    
    all_results = {}
    
    # Process each campaign
    campaign_iterator = tqdm(campaigns, desc="Analyzing campaign creativity") if show_progress else campaigns
    
    for campaign_id in campaign_iterator:
        try:
            # Load single campaign data
            campaign_data = {campaign_id: all_data[campaign_id]}
            df = load_dnd_data(campaign_data)
            
            if len(df) == 0:
                continue
            
            campaign_results = {}
            
            # Get embeddings for all text
            embeddings = get_embeddings(df)
            campaign_results['embeddings'] = embeddings
            
            # Calculate semantic distances
            semantic_distances = semantic_distance(df, embeddings=embeddings)
            campaign_results['semantic_distances'] = semantic_distances
            
            # Analyze session novelty
            novelty_results = session_novelty(df, embeddings=embeddings)
            campaign_results['session_novelty'] = novelty_results
            
            # Topic modeling
            try:
                topics_series, topic_model_obj = topic_model(df, embeddings=embeddings)
                campaign_results['topic_model'] = {
                    'topics': topics_series,
                    'model': topic_model_obj
                }
                
                # Add topics to DataFrame for transition analysis
                df_with_topics = df.copy()
                df_with_topics['topic'] = topics_series
                
                # Topic transitions
                topic_transitions = topic_transition_matrix(df_with_topics, topic_col='topic')
                campaign_results['topic_transitions'] = topic_transitions
                
                # Topic change rate
                change_rate = topic_change_rate(df_with_topics, topic_col='topic')
                campaign_results['topic_change_rate'] = {'overall_rate': change_rate.mean(), 'series': change_rate}
                
            except Exception as e:
                if show_progress:
                    print(f"Topic modeling failed for campaign {campaign_id}: {e}")
                campaign_results['topic_model'] = None
                campaign_results['topic_transitions'] = None
                campaign_results['topic_change_rate'] = None
            
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
            
            all_results[campaign_id] = campaign_results
            
        except Exception as e:
            if show_progress:
                print(f"Error processing campaign {campaign_id}: {e}")
            continue
    
    # Cache results
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(all_results, f)
        if show_progress:
            print(f"Cached results saved to {cache_file}")
    except Exception as e:
        if show_progress:
            print(f"Failed to save cache: {e}")
    
    return all_results


def aggregate_creativity_metrics(all_creativity_results: Dict) -> Dict:
    """
    Combine creativity results from multiple campaigns for statistical analysis.
    
    Parameters
    ----------
    all_creativity_results : Dict
        Output from analyze_creativity_all_campaigns()
        
    Returns
    -------
    Dict
        Aggregated statistics and comparisons across campaigns
    """
    import numpy as np
    import pandas as pd
    
    aggregated = {
        'campaign_summaries': {},
        'cross_campaign_stats': {},
        'distributions': {}
    }
    
    # Collect metrics from all campaigns
    semantic_distances_all = []
    novelty_scores_all = []
    topic_change_rates_all = []
    campaign_sizes = []
    campaign_ids = []
    
    # Label-aware metrics
    label_semantic_distances = {'in-character': [], 'out-of-character': [], 'mixed': []}
    label_message_counts = {'in-character': [], 'out-of-character': [], 'mixed': []}
    
    for campaign_id, results in all_creativity_results.items():
        if results is None:
            continue
            
        campaign_ids.append(campaign_id)
        
        # Semantic distances
        if 'semantic_distances' in results and results['semantic_distances'] is not None:
            distances = results['semantic_distances']
            mean_distance = np.mean(distances)
            semantic_distances_all.append(mean_distance)
        else:
            semantic_distances_all.append(np.nan)
        
        # Session novelty
        if 'session_novelty' in results and results['session_novelty'] is not None:
            novelty = results['session_novelty']
            if isinstance(novelty, dict) and 'novelty_scores' in novelty:
                avg_novelty = np.mean(novelty['novelty_scores'])
                novelty_scores_all.append(avg_novelty)
            else:
                novelty_scores_all.append(np.nan)
        else:
            novelty_scores_all.append(np.nan)
        
        # Topic change rates
        if 'topic_change_rate' in results and results['topic_change_rate'] is not None:
            change_rate = results['topic_change_rate']
            if isinstance(change_rate, dict) and 'overall_rate' in change_rate:
                topic_change_rates_all.append(change_rate['overall_rate'])
            else:
                topic_change_rates_all.append(np.nan)
        else:
            topic_change_rates_all.append(np.nan)
        
        # Campaign metadata
        if 'metadata' in results:
            campaign_sizes.append(results['metadata'].get('total_messages', 0))
        else:
            campaign_sizes.append(0)
        
        # Label-aware semantic distances
        if 'label_semantic_distances' in results:
            label_results = results['label_semantic_distances']
            for label in ['in-character', 'out-of-character', 'mixed']:
                if label in label_results and len(label_results[label]) > 0:
                    mean_distance = np.mean(label_results[label])
                    label_semantic_distances[label].append(mean_distance)
                    label_message_counts[label].append(len(label_results[label]))
                else:
                    label_semantic_distances[label].append(np.nan)
                    label_message_counts[label].append(0)
        else:
            # No label data available
            for label in ['in-character', 'out-of-character', 'mixed']:
                label_semantic_distances[label].append(np.nan)
                label_message_counts[label].append(0)
        
        # Individual campaign summary
        label_summary = {}
        for label in ['in-character', 'out-of-character', 'mixed']:
            label_summary[f'{label.replace("-", "_")}_semantic_distance'] = (
                label_semantic_distances[label][-1] if not np.isnan(label_semantic_distances[label][-1]) else None
            )
            label_summary[f'{label.replace("-", "_")}_message_count'] = label_message_counts[label][-1]
        
        aggregated['campaign_summaries'][campaign_id] = {
            'total_messages': results.get('metadata', {}).get('total_messages', 0),
            'unique_players': results.get('metadata', {}).get('unique_players', 0),
            'avg_semantic_distance': semantic_distances_all[-1] if not np.isnan(semantic_distances_all[-1]) else None,
            'avg_novelty_score': novelty_scores_all[-1] if not np.isnan(novelty_scores_all[-1]) else None,
            'topic_change_rate': topic_change_rates_all[-1] if not np.isnan(topic_change_rates_all[-1]) else None,
            'date_range': results.get('metadata', {}).get('date_range'),
            **label_summary
        }
    
    # Cross-campaign statistics
    valid_semantic = [x for x in semantic_distances_all if not np.isnan(x)]
    valid_novelty = [x for x in novelty_scores_all if not np.isnan(x)]
    valid_change_rates = [x for x in topic_change_rates_all if not np.isnan(x)]
    
    if valid_semantic:
        aggregated['cross_campaign_stats']['semantic_distance'] = {
            'mean': np.mean(valid_semantic),
            'std': np.std(valid_semantic),
            'min': np.min(valid_semantic),
            'max': np.max(valid_semantic),
            'median': np.median(valid_semantic),
            'campaigns_analyzed': len(valid_semantic)
        }
    
    if valid_novelty:
        aggregated['cross_campaign_stats']['novelty_score'] = {
            'mean': np.mean(valid_novelty),
            'std': np.std(valid_novelty),
            'min': np.min(valid_novelty),
            'max': np.max(valid_novelty),
            'median': np.median(valid_novelty),
            'campaigns_analyzed': len(valid_novelty)
        }
    
    if valid_change_rates:
        aggregated['cross_campaign_stats']['topic_change_rate'] = {
            'mean': np.mean(valid_change_rates),
            'std': np.std(valid_change_rates),
            'min': np.min(valid_change_rates),
            'max': np.max(valid_change_rates),
            'median': np.median(valid_change_rates),
            'campaigns_analyzed': len(valid_change_rates)
        }
    
    # Label-aware cross-campaign statistics
    for label in ['in-character', 'out-of-character', 'mixed']:
        valid_label_distances = [x for x in label_semantic_distances[label] if not np.isnan(x)]
        valid_label_counts = [count for i, count in enumerate(label_message_counts[label]) 
                             if not np.isnan(label_semantic_distances[label][i]) and count > 0]
        
        if valid_label_distances:
            label_key = f'{label.replace("-", "_")}_semantic_distance'
            aggregated['cross_campaign_stats'][label_key] = {
                'mean': np.mean(valid_label_distances),
                'std': np.std(valid_label_distances),
                'min': np.min(valid_label_distances),
                'max': np.max(valid_label_distances),
                'median': np.median(valid_label_distances),
                'campaigns_analyzed': len(valid_label_distances),
                'total_messages_analyzed': sum(valid_label_counts)
            }
    
    # Distribution data for plotting
    label_distributions = {}
    for label in ['in-character', 'out-of-character', 'mixed']:
        valid_label_distances = [x for x in label_semantic_distances[label] if not np.isnan(x)]
        label_key = f'{label.replace("-", "_")}_semantic_distances'
        label_distributions[label_key] = valid_label_distances
    
    aggregated['distributions'] = {
        'semantic_distances': valid_semantic,
        'novelty_scores': valid_novelty,
        'topic_change_rates': valid_change_rates,
        'campaign_sizes': campaign_sizes,
        'campaign_ids': campaign_ids,
        **label_distributions
    }
    
    # Overall summary
    aggregated['summary'] = {
        'total_campaigns_analyzed': len(campaign_ids),
        'campaigns_with_semantic_analysis': len(valid_semantic),
        'campaigns_with_novelty_analysis': len(valid_novelty),
        'campaigns_with_topic_analysis': len(valid_change_rates),
        'total_messages_across_all_campaigns': sum(campaign_sizes)
    }
    
    return aggregated