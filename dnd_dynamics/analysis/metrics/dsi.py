"""
Divergent Semantic Integration (DSI) Analysis

This module provides BERT-based DSI calculation for measuring semantic diversity
in D&D campaign text. Based on Johnson et al. (2023) methodology using layers 6 and 7.

Optimizations:
- GPU/MPS acceleration (CUDA for NVIDIA, MPS for Apple Silicon)
- Batched inference for processing multiple scenes in a single forward pass
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizer, BertModel
import torch

from . import _cache


# Global cache for BERT models to avoid reloading
_bert_model_cache = {}


def _get_device():
    """Get best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def clear_bert_cache():
    """Clear the BERT model cache to free memory."""
    global _bert_model_cache
    _bert_model_cache.clear()
    print("BERT model cache cleared")


def _get_bert_model(model_name="bert-base-uncased"):
    """Get or load BERT model with device placement."""
    if model_name not in _bert_model_cache:
        device = _get_device()
        print(f"Loading BERT model {model_name} on {device} (first time only)...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()
        model = model.to(device)
        _bert_model_cache[model_name] = {
            'tokenizer': tokenizer,
            'model': model,
            'device': device
        }
        print(f"BERT model {model_name} loaded and cached on {device}")

    return _bert_model_cache[model_name]


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
        if isinstance(message, dict):
            text = message.get('text', '')
        else:
            text = getattr(message, 'text', '')

        message_words = len(text.split()) if text else 0

        if current_word_count + message_words > target_words and current_scene:
            scenes.append(current_scene)
            current_scene = [message]
            current_word_count = message_words
        else:
            current_scene.append(message)
            current_word_count += message_words

    if current_scene:
        scenes.append(current_scene)

    return scenes


def _compute_dsi_from_embeddings(word_embeddings):
    """
    Compute DSI score from word embeddings.

    Args:
        word_embeddings: numpy array of shape (num_tokens, hidden_size)

    Returns:
        float: DSI score (average cosine distance)
    """
    if word_embeddings.shape[0] < 2:
        return None

    cosine_sim_matrix = cosine_similarity(word_embeddings)
    n = cosine_sim_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    cosine_similarities = cosine_sim_matrix[upper_triangle_indices]
    cosine_distances = 1 - cosine_similarities
    return float(np.mean(cosine_distances))


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

    cache = _get_bert_model(model_name)
    tokenizer = cache['tokenizer']
    model = cache['model']
    device = cache['device']

    tokens = tokenizer.tokenize(text)

    if len(tokens) < 5:
        return None

    max_tokens = 512 - 2
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        hidden_states = outputs.hidden_states

    layer_6 = hidden_states[6][0]
    layer_7 = hidden_states[7][0]
    combined_embeddings = (layer_6 + layer_7) / 2
    word_embeddings = combined_embeddings[1:-1].cpu().numpy()

    return _compute_dsi_from_embeddings(word_embeddings)


def calculate_dsi_bert_batch(texts: List[str], model_name: str = "bert-base-uncased") -> List[Optional[float]]:
    """
    Calculate DSI for multiple texts in a single batched forward pass.

    Args:
        texts: List of text strings to analyze
        model_name: BERT model to use

    Returns:
        List of DSI scores (None for texts that couldn't be analyzed)
    """
    if not texts:
        return []

    cache = _get_bert_model(model_name)
    tokenizer = cache['tokenizer']
    model = cache['model']
    device = cache['device']

    # Tokenize all texts and track which are valid
    all_tokens = []
    valid_indices = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 5:
            continue
        max_tokens = 512 - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        all_tokens.append(tokens)
        valid_indices.append(i)

    if not all_tokens:
        return [None] * len(texts)

    # Convert to input IDs and pad to same length
    all_input_ids = []
    for tokens in all_tokens:
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        all_input_ids.append(input_ids)

    max_len = max(len(ids) for ids in all_input_ids)
    padded_input_ids = []
    attention_masks = []

    for input_ids in all_input_ids:
        padding_length = max_len - len(input_ids)
        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * padding_length)
        attention_masks.append([1] * len(input_ids) + [0] * padding_length)

    input_tensor = torch.tensor(padded_input_ids).to(device)
    attention_tensor = torch.tensor(attention_masks).to(device)

    # Run batched inference
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attention_tensor)
        hidden_states = outputs.hidden_states

    # Extract layers 6 and 7 for all items in batch
    layer_6 = hidden_states[6]
    layer_7 = hidden_states[7]
    combined = (layer_6 + layer_7) / 2

    # Calculate DSI for each item
    results = [None] * len(texts)

    for batch_idx, text_idx in enumerate(valid_indices):
        seq_len = len(all_input_ids[batch_idx])
        word_embeddings = combined[batch_idx, 1:seq_len-1].cpu().numpy()
        results[text_idx] = _compute_dsi_from_embeddings(word_embeddings)

    return results


def _analyze_single_campaign_dsi(df: pd.DataFrame, campaign_id: str,
                                  target_words: int = 175,
                                  batch_size: int = 16,
                                  show_progress: bool = False) -> Optional[Dict]:
    """
    Run DSI analysis for a single campaign using batched inference.

    Args:
        df: Campaign DataFrame with 'text' column
        campaign_id: Campaign identifier
        target_words: Words per scene for DSI calculation
        batch_size: Number of scenes to process in each batch
        show_progress: Whether to show progress indicators

    Returns:
        Dict with DSI analysis results or None if analysis failed
    """
    messages = [{'text': row['text'], 'message_id': idx} for idx, row in df.iterrows()]
    scenes = create_word_scenes(messages, target_words=target_words)

    # Prepare scene texts
    scene_texts = []
    scene_word_counts = []
    for scene in scenes:
        scene_text = ' '.join([msg['text'] for msg in scene if msg.get('text')])
        scene_texts.append(scene_text)
        scene_word_counts.append(len(scene_text.split()) if scene_text else 0)

    # Process in batches
    scene_dsi_scores = []
    for i in range(0, len(scene_texts), batch_size):
        batch = scene_texts[i:i + batch_size]
        batch_scores = calculate_dsi_bert_batch(batch)
        scene_dsi_scores.extend(batch_scores)

    # Convert None to np.nan for statistics
    scores_for_stats = [s if s is not None else np.nan for s in scene_dsi_scores]
    valid_scores = [s for s in scene_dsi_scores if s is not None]
    time_averaged_dsi = np.mean(valid_scores) if valid_scores else np.nan

    return {
        'scene_dsi_scores': scores_for_stats,
        'scene_word_counts': scene_word_counts,
        'time_averaged_dsi': time_averaged_dsi,
        'scene_count': len(scenes),
        'metadata': {
            'campaign_id': campaign_id,
            'total_messages': len(df),
        }
    }


def analyze_dsi(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                target_words: int = 175,
                batch_size: int = 16,
                show_progress: bool = True,
                cache_dir: Optional[str] = None,
                force_refresh: bool = False) -> Union[Dict, Dict[str, Dict]]:
    """
    Calculate DSI metrics for single or multiple campaigns using DataFrames.

    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        target_words: Words per scene for DSI calculation
        batch_size: Number of scenes to process in each forward pass
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/dsi_results)
        force_refresh: Whether to force recomputation even if cached results exist

    Returns:
        Dict of DSI results for single campaign, or Dict[campaign_id, results] for multiple
    """
    if isinstance(data, pd.DataFrame):
        return _analyze_single_campaign_dsi(data, "not specified",
                                            target_words=target_words,
                                            batch_size=batch_size,
                                            show_progress=False)

    elif isinstance(data, dict):
        if cache_dir is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            cache_dir = str(repo_root / 'data' / 'processed' / 'dsi_results')

        cached_results, data_to_process = _cache.handle_multi_campaign_caching(
            data, cache_dir, force_refresh, show_progress, "DSI"
        )

        new_results = {}
        if data_to_process:
            if show_progress and len(data_to_process) > 1:
                iterator = tqdm(data_to_process.items(), desc="Analyzing campaign DSI", total=len(data_to_process))
            else:
                iterator = data_to_process.items()

            for campaign_id, df in iterator:
                new_results[campaign_id] = _analyze_single_campaign_dsi(
                    df, campaign_id, target_words=target_words,
                    batch_size=batch_size, show_progress=False
                )

        return _cache.save_new_results_and_combine(
            cached_results, new_results, cache_dir, show_progress, "DSI"
        )

    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.DataFrame or Dict[str, pd.DataFrame]")
