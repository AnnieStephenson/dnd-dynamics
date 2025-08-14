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
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from gensim import corpora, models
from gensim.utils import simple_preprocess
from transformers import BertTokenizer, BertModel
import torch
import os
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Import other modules using relative imports
from . import data_loading as dl
from . import batch


# ===================================================================
# SENTENCE EMBEDDING DISTANCE ANALYSIS (SBERT)
# ===================================================================


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
        repo_root = Path(__file__).parent.parent
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
        repo_root = Path(__file__).parent.parent
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

# ===================================================================
# TOPIC MODELING ANALYSIS
# ===================================================================

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
        return _topic_model_bertopic(texts, n_topics, embeddings, random_state)

    elif engine.lower() == "lda":
        return _topic_model_lda(texts, n_topics, random_state)

    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'bertopic' or 'lda'")


def _topic_model_bertopic(texts: List[str],
                         n_topics: int,
                         embeddings: Optional[np.ndarray],
                         random_state: int) -> Tuple[pd.Series, 'BERTopic']:
    """BERTopic implementation."""

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

# ===================================================================
# DIVERGENT SEMANTIC INTEGRATION (DSI) ANALYSIS (BERT)
# ===================================================================

def create_word_scenes(messages, target_words=175):
    """
    Segment campaign messages into scenes of approximately target_words length.
    Never split individual messages - only complete messages per scene.
    
    Args:
        messages: List of message dictionaries with 'text' field
        target_words: Target word count per scene (default 175)
    
    Returns:
        List of scenes, where each scene is a list of complete messages
    """
    scenes = []
    current_scene = []
    current_word_count = 0

    for message in messages:
        # Get text content from message
        if isinstance(message, dict):
            text = message.get('text', '')
        else:
            # Handle DataFrame row
            text = getattr(message, 'text', '')

        # Count words in this message
        message_words = len(text.split()) if text else 0

        # If adding this message would exceed target and we have some content, start new scene
        if current_word_count + message_words > target_words and current_scene:
            scenes.append(current_scene)
            current_scene = [message]
            current_word_count = message_words
        else:
            # Add message to current scene
            current_scene.append(message)
            current_word_count += message_words

    # Add the last scene if it has content
    if current_scene:
        scenes.append(current_scene)

    return scenes

# Global cache for BERT models to avoid reloading
_bert_model_cache = {}

def clear_bert_cache():
    """Clear the BERT model cache to free memory."""
    global _bert_model_cache
    _bert_model_cache.clear()
    print("🗑️  BERT model cache cleared")


def calculate_dsi_bert(text, model_name="bert-base-uncased"):
    """
    Calculate Divergent Semantic Integration using BERT embeddings.
    Based on Johnson et al. (2023) methodology using layers 6 and 7.
    
    Args:
        text: Input text string (scene text)
        model_name: BERT model to use
        
    Returns:
        float: DSI score (average cosine distance between word embeddings)
    """
    if not text or not text.strip():
        return None

    # Initialize BERT model and tokenizer (with caching)
    if model_name not in _bert_model_cache:
        try:
            print(f"Loading BERT model {model_name} (first time only)...")
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name, output_hidden_states=True)
            model.eval()
            _bert_model_cache[model_name] = {'tokenizer': tokenizer, 'model': model}
            print(f"✅ BERT model {model_name} loaded and cached")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            return None

    tokenizer = _bert_model_cache[model_name]['tokenizer']
    model = _bert_model_cache[model_name]['model']

    # Tokenize input text
    tokens = tokenizer.tokenize(text)

    # Skip if too few tokens for meaningful analysis
    if len(tokens) < 5:
        return None

    # Handle long sequences by truncating
    max_tokens = 512 - 2  # Account for [CLS] and [SEP] tokens
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    # Convert to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_tensor = torch.tensor([input_ids])

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        hidden_states = outputs.hidden_states

    # Extract embeddings from layers 6 and 7 (following Johnson et al. methodology)
    layer_6 = hidden_states[6][0]  # Shape: (seq_len, hidden_size)
    layer_7 = hidden_states[7][0]  # Shape: (seq_len, hidden_size)

    # Average layers 6 and 7
    combined_embeddings = (layer_6 + layer_7) / 2

    # Remove [CLS] and [SEP] tokens, keep only actual word embeddings
    word_embeddings = combined_embeddings[1:-1]  # Shape: (num_tokens, hidden_size)

    # Skip if too few word embeddings
    if word_embeddings.shape[0] < 2:
        return None

    # Calculate pairwise cosine similarities
    embeddings_np = word_embeddings.numpy()
    cosine_sim_matrix = cosine_similarity(embeddings_np)

    # Extract upper triangle (excluding diagonal) to avoid double counting
    n = cosine_sim_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    cosine_similarities = cosine_sim_matrix[upper_triangle_indices]

    # Convert to cosine distances (distance = 1 - similarity)
    cosine_distances = 1 - cosine_similarities

    # Return average cosine distance as DSI score
    dsi_score = np.mean(cosine_distances)

    return float(dsi_score)


def analyze_campaign_dsi_over_time(campaign_data):
    """
    Calculate DSI scores for each scene in a campaign over time.
    
    Args:
        campaign_data: Campaign message data (dict with message IDs as keys)
        
    Returns:
        dict: {
            'scene_dsi_scores': [list of DSI scores],
            'scene_word_counts': [list of word counts per scene],
            'time_averaged_dsi': float,
            'scene_count': int
        }
    """
    # Convert campaign data to list of messages
    messages = []
    for message_id, message_info in campaign_data.items():
        # Extract text from different possible formats
        text = ''
        if 'text' in message_info:
            text = message_info['text']
        elif 'paragraphs' in message_info:
            # Combine paragraph text
            paragraphs = message_info['paragraphs']
            text_segments = []
            for para_id in sorted(paragraphs.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                para_data = paragraphs[para_id]
                if isinstance(para_data, dict) and 'text' in para_data:
                    text_segments.append(para_data['text'])
            text = ' '.join(text_segments)

        messages.append({'text': text, 'message_id': message_id})

    # Create scenes
    scenes = create_word_scenes(messages, target_words=175)

    # Calculate DSI for each scene
    scene_dsi_scores = []
    scene_word_counts = []
    total_scenes = len(scenes)

    print(f"Processing {total_scenes} scenes for DSI analysis...")

    for i, scene in enumerate(scenes):
        # Progress indicator
        if i % 100 == 0:
            print(f"  Scene {i+1}/{total_scenes} ({100*(i+1)/total_scenes:.1f}%)")

        # Combine all text in the scene
        scene_text = ' '.join([msg['text'] for msg in scene if msg.get('text')])

        # Count words in scene
        word_count = len(scene_text.split()) if scene_text else 0
        scene_word_counts.append(word_count)

        # Calculate DSI score
        if scene_text.strip():
            dsi_score = calculate_dsi_bert(scene_text)
            if dsi_score is not None:
                scene_dsi_scores.append(dsi_score)
            else:
                scene_dsi_scores.append(np.nan)
        else:
            scene_dsi_scores.append(np.nan)

    print(f"✅ Completed DSI analysis for {total_scenes} scenes")

    # Calculate time-averaged DSI (excluding NaN values)
    valid_scores = [score for score in scene_dsi_scores if not np.isnan(score)]
    time_averaged_dsi = np.mean(valid_scores) if valid_scores else np.nan

    return {
        'scene_dsi_scores': scene_dsi_scores,
        'scene_word_counts': scene_word_counts,
        'time_averaged_dsi': time_averaged_dsi,
        'scene_count': len(scenes),
        'valid_scores_count': len(valid_scores)
    }


def calculate_campaign_average_dsi(campaign_data):
    """
    Calculate time-averaged DSI for entire campaign.
    
    Args:
        campaign_data: Campaign message data
        
    Returns:
        float: Average DSI score across all scenes
    """
    analysis = analyze_campaign_dsi_over_time(campaign_data)
    return analysis['time_averaged_dsi']



def _analyze_single_campaign_dsi(df: pd.DataFrame, campaign_id: str, target_words: int = 175, show_progress: bool = False) -> Optional[Dict]:
    """
    Run DSI analysis for a single campaign using DataFrame.
    
    Args:
        df: Campaign DataFrame with 'text' column
        campaign_id: Campaign identifier
        target_words: Words per scene for DSI calculation
        show_progress: Whether to show progress indicators
        
    Returns:
        Dict with DSI analysis results or None if analysis failed
    """
    # Convert DataFrame to list of messages for scene creation
    messages = []
    for idx, row in df.iterrows():
        messages.append({'text': row['text'], 'message_id': idx})

    # Create scenes
    scenes = create_word_scenes(messages, target_words=target_words)

    # Calculate DSI for each scene
    scene_dsi_scores = []
    scene_word_counts = []
    total_scenes = len(scenes)
    
    for i, scene in enumerate(scenes):
        # Combine all message text in this scene
        scene_text = ' '.join([msg['text'] for msg in scene if msg.get('text')])
        
        if scene_text.strip():
            # Calculate DSI score
            scene_dsi = calculate_dsi_bert(scene_text)
            scene_dsi_scores.append(scene_dsi)
            
            # Count words in scene
            word_count = len(scene_text.split())
            scene_word_counts.append(word_count)
        else:
            scene_dsi_scores.append(np.nan)
            scene_word_counts.append(0)

    # Calculate time-averaged DSI (excluding NaN values)
    valid_scores = [score for score in scene_dsi_scores if not np.isnan(score)]
    time_averaged_dsi = np.mean(valid_scores) if valid_scores else np.nan

    dsi_analysis = {
        'scene_dsi_scores': scene_dsi_scores,
        'scene_word_counts': scene_word_counts,
        'time_averaged_dsi': time_averaged_dsi,
        'scene_count': total_scenes
    }

    # Add campaign metadata
    dsi_analysis['metadata'] = {
        'campaign_id': campaign_id,
        'total_messages': len(df),
    }

    return dsi_analysis




# ===================================================================
# Multi-metric ANALYSIS 
# ===================================================================

def _analyze_single_campaign_creativity(df, campaign_id: str, show_progress: bool) -> Dict:
    """
    Run all creativity analysis functions for a single campaign.
    
    Args:
        df: Loaded campaign DataFrame
        campaign_id: Campaign identifier
        show_progress: Whether to show progress indicators
        
    Returns:
        Dict with creativity analysis results
    """
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

# ===================================================================
# User functions
# ===================================================================

def analyze_creativity(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Union[Dict, Dict[str, Dict]]:
    """
    Analyze creativity metrics for single or multiple campaigns.
    
    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/creativity_results)
        force_refresh: Whether to force recomputation even if cached results exist
        
    Returns:
        Dict of results for single campaign, or Dict[campaign_id, results] for multiple
    """
    if isinstance(data, pd.DataFrame):
        # Single campaign analysis - no caching for single campaigns
        campaign_id = "not specified"
        return _analyze_single_campaign_creativity(data, campaign_id, show_progress=False)
    
    elif isinstance(data, dict):
        # Multiple campaign analysis with caching support
        
        # Set default cache directory
        if cache_dir is None:
            repo_root = Path(__file__).parent.parent
            cache_dir = str(repo_root / 'data' / 'processed' / 'creativity_results')
        
        # Handle caching using helper function
        cached_results, data_to_process = batch.handle_multi_campaign_caching(
            data, cache_dir, force_refresh, show_progress, "creativity"
        )
        
        # Process missing campaigns
        new_results = {}
        if data_to_process:
            if show_progress and len(data_to_process) > 1:
                iterator = tqdm(data_to_process.items(), desc="Analyzing creativity", total=len(data_to_process))
            else:
                iterator = data_to_process.items()
            
            for campaign_id, df in iterator:
                new_results[campaign_id] = _analyze_single_campaign_creativity(
                    df, campaign_id, show_progress=False
                )
        
        # Save new results and combine with cached results
        return batch.save_new_results_and_combine(
            cached_results, new_results, cache_dir, show_progress, "creativity"
        )
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.DataFrame or Dict[str, pd.DataFrame]")


def analyze_dsi(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], target_words: int = 175,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Union[Dict, Dict[str, Dict]]:
    """
    Calculate DSI metrics for single or multiple campaigns using DataFrames.
    
    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        target_words: Words per scene for DSI calculation
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/dsi_results)
        force_refresh: Whether to force recomputation even if cached results exist
        
    Returns:
        Dict of DSI results for single campaign, or Dict[campaign_id, results] for multiple
    """
    if isinstance(data, pd.DataFrame):
        # Single campaign analysis - no caching for single campaigns
        campaign_id = "not specified"
        return _analyze_single_campaign_dsi(data, campaign_id, target_words=target_words, show_progress=False)
    
    elif isinstance(data, dict):
        # Multiple campaign analysis with caching support
        
        # Set default cache directory
        if cache_dir is None:
            repo_root = Path(__file__).parent.parent
            cache_dir = str(repo_root / 'data' / 'processed' / 'dsi_results')
        
        # Handle caching using helper function
        cached_results, data_to_process = batch.handle_multi_campaign_caching(
            data, cache_dir, force_refresh, show_progress, "DSI"
        )
        
        # Process missing campaigns
        new_results = {}
        if data_to_process:
            if show_progress and len(data_to_process) > 1:
                iterator = tqdm(data_to_process.items(), desc="Analyzing campaign DSI", total=len(data_to_process))
            else:
                iterator = data_to_process.items()
            
            for campaign_id, df in iterator:
                new_results[campaign_id] = _analyze_single_campaign_dsi(
                    df, campaign_id, target_words=target_words, show_progress=False
                )
        
        # Save new results and combine with cached results
        return batch.save_new_results_and_combine(
            cached_results, new_results, cache_dir, show_progress, "DSI"
        )
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.DataFrame or Dict[str, pd.DataFrame]")
