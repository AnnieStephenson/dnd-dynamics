"""
Multi-campaign batch analysis functions for D&D gameplay data.
Provides tools for analyzing creativity metrics, DSI scores, and campaign statistics across large datasets with intelligent caching.
"""

import json
import os
import pickle
import hashlib
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import other modules using relative imports
from . import data_loading as dl
from . import basic_metrics as basic
from . import creativity_metrics as creativity


# ===================================================================
# MULTI-METRIC analysis for single campaign
# ===================================================================


def _analyze_creativity_from_individual_files(campaigns_dir: Path, max_campaigns: Optional[int],
                                           cache_dir: str, force_refresh: bool, show_progress: bool) -> Dict:
    """Memory-efficient creativity analysis using individual campaign files."""
    
    campaign_files = sorted([f for f in campaigns_dir.glob('*.json') if f.is_file()])
    campaigns_to_analyze = min(max_campaigns, len(campaign_files)) if max_campaigns is not None else len(campaign_files)
    
    # Cache handling
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = Path(cache_dir) / f'creativity_analysis_{campaigns_to_analyze}_campaigns.pkl'
    
    if not force_refresh and cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Process campaigns
    files_to_process = campaign_files[:campaigns_to_analyze]
    iterator = tqdm(files_to_process, desc="Analyzing campaigns") if show_progress else files_to_process
    all_results = {}
    
    for campaign_file in iterator:
        campaign_id = campaign_file.stem
        
        with open(campaign_file, 'r', encoding='utf-8') as f:
            campaign_data = json.load(f)
        
        single_campaign_data = {campaign_id: campaign_data}
        df = dl._load_dnd_data(single_campaign_data)
        
        campaign_results = _analyze_single_campaign_creativity(df, campaign_id, campaign_data, show_progress)
        all_results[campaign_id] = campaign_results
    
    # Cache results
    with open(cache_file, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return all_results


def _analyze_single_campaign_creativity(df, campaign_id: str, campaign_data: dict, show_progress: bool) -> Dict:
    """
    Run all creativity analysis functions for a single campaign.
    
    Args:
        df: Loaded campaign DataFrame
        campaign_id: Campaign identifier
        campaign_data: Raw campaign data
        show_progress: Whether to show progress indicators
        
    Returns:
        Dict with creativity analysis results
    """
    campaign_results = {}
    
    # Get embeddings for all text
    embeddings = creativity.get_embeddings(df)
    campaign_results['embeddings'] = embeddings
    
    # Calculate semantic distances
    semantic_distances = creativity.semantic_distance(df, embeddings=embeddings)
    campaign_results['semantic_distances'] = semantic_distances
    
    # Analyze session novelty
    novelty_results = creativity.session_novelty(df, embeddings=embeddings)
    campaign_results['session_novelty'] = novelty_results
    
    # Topic modeling
    topics_series, topic_model_obj = creativity.topic_model(df, embeddings=embeddings)
    campaign_results['topic_model'] = {
        'topics': topics_series,
        'model': topic_model_obj
    }
    
    # Add topics to DataFrame for transition analysis
    df_with_topics = df.copy()
    df_with_topics['topic'] = topics_series
    
    # Topic transitions
    topic_transitions = creativity.topic_transition_matrix(df_with_topics, topic_col='topic')
    campaign_results['topic_transitions'] = topic_transitions
    
    # Topic change rate
    change_rate = creativity.topic_change_rate(df_with_topics, topic_col='topic')
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
# MULTI-CAMPAIGN analysis
# ===================================================================

def analyze_all_campaigns(campaign_dataframes: Dict[str, pd.DataFrame], 
                         original_json_data: Optional[Dict] = None,
                         show_progress: bool = True) -> Dict[str, Dict]:
    """
    Apply all analysis functions across multiple campaigns.
    
    Args:
        campaign_dataframes: Dictionary of campaign DataFrames from load_campaigns()
        original_json_data: Optional original JSON data for paragraph-level action analysis
        show_progress: Whether to show progress indicators
        
    Returns:
        Dict containing per-campaign and aggregated results for all metrics
    """
    results = {
        'per_campaign': {},
        'aggregated': {},
        'summary_stats': {}
    }
    
    iterator = tqdm(campaign_dataframes.items(), desc="Analyzing campaigns") if show_progress else campaign_dataframes.items()
    
    for campaign_id, df in iterator:
        # Run all analysis functions for this campaign
        campaign_results = {
            'time_intervals_overall': basic.analyze_time_intervals(df, by_player=False),
            'time_intervals_by_player': basic.analyze_time_intervals(df, by_player=True),
            'cumulative_posts_overall': basic.analyze_cumulative_posts(df, by_player=False),
            'cumulative_posts_by_player': basic.analyze_cumulative_posts(df, by_player=True),
            'unique_players_characters': basic.analyze_unique_players_characters(df),
            'post_lengths_overall': basic.analyze_post_lengths(df, by_player=False),
            'post_lengths_by_player': basic.analyze_post_lengths(df, by_player=True),
            'post_lengths_by_label_overall': basic.analyze_post_lengths_by_label(df, by_player=False),
            'post_lengths_by_label_by_player': basic.analyze_post_lengths_by_label(df, by_player=True),
            'character_mentions': basic.analyze_character_mentions(df),
            'dice_roll_frequency': basic.analyze_dice_roll_frequency(df),
            'summary_report': basic.generate_summary_report(df)
        }
        
        # Add paragraph-level action analysis if original JSON data is available
        if original_json_data and campaign_id in original_json_data:
            single_campaign_data = {campaign_id: original_json_data[campaign_id]}
            campaign_results['paragraph_actions'] = basic.analyze_paragraph_actions(single_campaign_data)
        
        results['per_campaign'][campaign_id] = campaign_results
    
    # Generate aggregated results
    results['aggregated'] = aggregate_campaign_metrics(results['per_campaign'])
    results['summary_stats'] = generate_multi_campaign_summary(campaign_dataframes, results['per_campaign'])
    
    return results

def analyze_creativity_all_campaigns(data_file_path: Optional[str] = None,
                                   max_campaigns: Optional[int] = None,
                                   cache_dir: Optional[str] = None,
                                   force_refresh: bool = False,
                                   show_progress: bool = True) -> Dict:
    """
    Apply all existing creative metric functions across multiple campaigns.
    
    MEMORY OPTIMIZED: Loads individual campaign files instead of entire JSON file.
    
    Parameters
    ----------
    data_file_path : str, optional
        Path to campaigns directory or legacy JSON file (auto-detects)
        If None, defaults to 'data/raw-human-games/individual_campaigns' relative to repo root
    max_campaigns : int, optional
        Maximum number of campaigns to analyze (None for all)
    cache_dir : str, optional
        Directory to store cached results. If None, defaults to 'data/processed/creativity_cache' relative to repo root
    force_refresh : bool
        Whether to force recalculation even if cache exists
    show_progress : bool
        Whether to show progress bars
        
    Returns
    -------
    Dict
        Results from all creative metrics functions for each campaign
    """
    # Set default paths relative to the repository root
    repo_root = Path(__file__).parent.parent
    if data_file_path is None:
        data_file_path = str(repo_root / 'data' / 'raw-human-games' / 'individual_campaigns')
    if cache_dir is None:
        cache_dir = str(repo_root / 'data' / 'processed' / 'creativity_cache')
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use individual campaign files directly
    return _analyze_creativity_from_individual_files(
        Path(data_file_path), max_campaigns, cache_dir, force_refresh, show_progress
    )

def analyze_dsi_all_campaigns(data_file_path: Optional[str] = None,
                            max_campaigns: Optional[int] = None,
                            show_progress: bool = True,
                            cache_dir: Optional[str] = None,
                            force_refresh: bool = False) -> Dict:
    """
    Calculate DSI metrics for multiple campaigns with memory-efficient loading.
    
    MEMORY OPTIMIZED: Loads individual campaign files instead of entire JSON file.
    
    Args:
        data_file_path: Path to campaigns directory or legacy JSON file (auto-detects)
                       If None, defaults to 'data/raw-human-games/individual_campaigns' relative to repo root
        max_campaigns: Maximum number of campaigns to process
        show_progress: Whether to show progress indicators
        cache_dir: Directory to store cache files
        force_refresh: Whether to force recalculation even if cache exists
        
    Returns:
        dict: Campaign IDs mapped to DSI analysis results
    """
    # Set default paths relative to the repository root
    repo_root = Path(__file__).parent.parent
    if data_file_path is None:
        data_file_path = str(repo_root / 'data' / 'raw-human-games' / 'individual_campaigns')
    if cache_dir is None:
        cache_dir = str(repo_root / 'data' / 'processed' / 'dsi_cache')
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Use individual campaign files directly
    return creativity._analyze_dsi_from_individual_files(
        Path(data_file_path), max_campaigns, cache_dir, force_refresh, show_progress
    )

def generate_multi_campaign_summary(campaign_dataframes: Dict[str, pd.DataFrame], 
                                   per_campaign_results: Dict[str, Dict]) -> Dict:
    """
    Generate summary statistics across all campaigns.
    
    Args:
        campaign_dataframes: Dictionary of campaign DataFrames
        per_campaign_results: Per-campaign analysis results
        
    Returns:
        Dict containing multi-campaign summary statistics
    """
    total_campaigns = len(campaign_dataframes)
    total_messages = sum(len(df) for df in campaign_dataframes.values())
    
    # Campaign size statistics
    campaign_sizes = [len(df) for df in campaign_dataframes.values()]
    
    # Duration statistics
    campaign_durations = []
    for df in campaign_dataframes.values():
        if len(df) > 1:
            duration = (df['date'].max() - df['date'].min()).days
            campaign_durations.append(duration)
    
    # Player count statistics
    player_counts = []
    for df in campaign_dataframes.values():
        player_counts.append(df['player'].nunique())
    
    # Activity statistics (posts per day)
    activity_rates = []
    for df in campaign_dataframes.values():
        if len(df) > 1:
            duration_days = (df['date'].max() - df['date'].min()).days
            if duration_days > 0:
                activity_rates.append(len(df) / duration_days)
    
    summary = {
        'total_campaigns': total_campaigns,
        'total_messages': total_messages,
        'campaign_size_stats': {
            'mean_messages': np.mean(campaign_sizes) if campaign_sizes else 0,
            'median_messages': np.median(campaign_sizes) if campaign_sizes else 0,
            'min_messages': np.min(campaign_sizes) if campaign_sizes else 0,
            'max_messages': np.max(campaign_sizes) if campaign_sizes else 0,
            'std_messages': np.std(campaign_sizes) if campaign_sizes else 0
        },
        'duration_stats': {
            'mean_days': np.mean(campaign_durations) if campaign_durations else 0,
            'median_days': np.median(campaign_durations) if campaign_durations else 0,
            'min_days': np.min(campaign_durations) if campaign_durations else 0,
            'max_days': np.max(campaign_durations) if campaign_durations else 0,
            'std_days': np.std(campaign_durations) if campaign_durations else 0
        },
        'player_count_stats': {
            'mean_players': np.mean(player_counts) if player_counts else 0,
            'median_players': np.median(player_counts) if player_counts else 0,
            'min_players': np.min(player_counts) if player_counts else 0,
            'max_players': np.max(player_counts) if player_counts else 0,
            'std_players': np.std(player_counts) if player_counts else 0
        },
        'activity_stats': {
            'mean_posts_per_day': np.mean(activity_rates) if activity_rates else 0,
            'median_posts_per_day': np.median(activity_rates) if activity_rates else 0,
            'min_posts_per_day': np.min(activity_rates) if activity_rates else 0,
            'max_posts_per_day': np.max(activity_rates) if activity_rates else 0,
            'std_posts_per_day': np.std(activity_rates) if activity_rates else 0
        }
    }
    
    return summary

# ===================================================================
# Aggregation functions
# ===================================================================

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

def aggregate_campaign_metrics(per_campaign_results: Dict[str, Dict]) -> Dict:
    """
    Aggregate metrics across all campaigns to create combined statistics.
    
    Args:
        per_campaign_results: Results from analyze_all_campaigns per_campaign section
        
    Returns:
        Dict containing aggregated metrics across all campaigns
    """
    aggregated = {}
    
    if not per_campaign_results:
        return aggregated
    
    campaign_ids = list(per_campaign_results.keys())
    
    # Aggregate time intervals
    all_intervals = []
    for campaign_id in campaign_ids:
        intervals = per_campaign_results[campaign_id]['time_intervals_overall']['overall']['intervals_data']
        all_intervals.extend(intervals)
    
    if all_intervals:
        all_intervals = np.array(all_intervals)
        aggregated['time_intervals'] = {
            'mean_hours': np.mean(all_intervals),
            'median_hours': np.median(all_intervals),
            'std_hours': np.std(all_intervals),
            'count': len(all_intervals),
            'intervals_data': all_intervals
        }
    
    # Aggregate post lengths
    all_word_counts = []
    for campaign_id in campaign_ids:
        word_counts = per_campaign_results[campaign_id]['post_lengths_overall']['overall']['word_counts_data']
        all_word_counts.extend(word_counts)
    
    if all_word_counts:
        all_word_counts = np.array(all_word_counts)
        aggregated['post_lengths'] = {
            'mean_words': np.mean(all_word_counts),
            'median_words': np.median(all_word_counts),
            'std_words': np.std(all_word_counts),
            'max_words': np.max(all_word_counts),
            'count': len(all_word_counts),
            'word_counts_data': all_word_counts
        }
    
    # Aggregate dice roll frequency
    total_posts_with_rolls = sum(per_campaign_results[cid]['dice_roll_frequency']['posts_with_rolls'] 
                               for cid in campaign_ids)
    total_all_posts = sum(per_campaign_results[cid]['dice_roll_frequency']['total_posts'] 
                         for cid in campaign_ids)
    
    if total_all_posts > 0:
        aggregated['dice_rolls'] = {
            'total_posts': total_all_posts,
            'posts_with_rolls': total_posts_with_rolls,
            'posts_without_rolls': total_all_posts - total_posts_with_rolls,
            'roll_percentage': (total_posts_with_rolls / total_all_posts) * 100
        }
    
    # Aggregate character mentions
    all_mentions = {}
    for campaign_id in campaign_ids:
        campaign_mentions = per_campaign_results[campaign_id]['character_mentions']['full_counts']
        for character, count in campaign_mentions.items():
            all_mentions[character] = all_mentions.get(character, 0) + count
    
    # Always include character_mentions in aggregated results, even if empty
    if all_mentions:
        # Sort by frequency and get top mentions
        sorted_mentions = dict(sorted(all_mentions.items(), key=lambda x: x[1], reverse=True))
        top_15_mentions = dict(list(sorted_mentions.items())[:15])
        
        aggregated['character_mentions'] = {
            'total_mentions': sum(all_mentions.values()),
            'unique_characters_mentioned': len(all_mentions),
            'top_mentions': top_15_mentions,
            'top_mentions_names': list(top_15_mentions.keys()),
            'top_mentions_counts': list(top_15_mentions.values()),
            'full_counts': all_mentions
        }
    else:
        # No character mentions found across any campaign
        aggregated['character_mentions'] = {
            'total_mentions': 0,
            'unique_characters_mentioned': 0,
            'top_mentions': {},
            'top_mentions_names': [],
            'top_mentions_counts': [],
            'full_counts': {}
        }
    
    # Aggregate paragraph-level action analysis (if available)
    paragraph_campaigns = [cid for cid in campaign_ids 
                          if 'paragraph_actions' in per_campaign_results[cid]]
    
    if paragraph_campaigns:
        # Sum all paragraph action counts across campaigns
        total_paragraph_actions = {
            'name_mentions_paragraphs': 0,
            'spells_paragraphs': 0,
            'dialogue_paragraphs': 0,
            'roll_paragraphs': 0,
            'weapon_paragraphs': 0,
            'no_action_paragraphs': 0,
            'total_paragraphs': 0,
            'in_character_paragraphs': 0,
            'out_of_character_paragraphs': 0,
            'mixed_paragraphs': 0,
            'unlabeled_paragraphs': 0
        }
        
        for campaign_id in paragraph_campaigns:
            para_data = per_campaign_results[campaign_id]['paragraph_actions']
            for key in total_paragraph_actions:
                total_paragraph_actions[key] += para_data.get(key, 0)
        
        # Calculate percentages if we have paragraphs
        if total_paragraph_actions['total_paragraphs'] > 0:
            total_paras = total_paragraph_actions['total_paragraphs']
            
            aggregated['paragraph_actions'] = {
                **total_paragraph_actions,
                # Action type percentages
                'name_mentions_percentage': (total_paragraph_actions['name_mentions_paragraphs'] / total_paras) * 100,
                'spells_percentage': (total_paragraph_actions['spells_paragraphs'] / total_paras) * 100,
                'dialogue_percentage': (total_paragraph_actions['dialogue_paragraphs'] / total_paras) * 100,
                'roll_percentage': (total_paragraph_actions['roll_paragraphs'] / total_paras) * 100,
                'weapon_percentage': (total_paragraph_actions['weapon_paragraphs'] / total_paras) * 100,
                'no_action_percentage': (total_paragraph_actions['no_action_paragraphs'] / total_paras) * 100,
                # Character label percentages
                'in_character_percentage': (total_paragraph_actions['in_character_paragraphs'] / total_paras) * 100,
                'out_of_character_percentage': (total_paragraph_actions['out_of_character_paragraphs'] / total_paras) * 100,
                'mixed_percentage': (total_paragraph_actions['mixed_paragraphs'] / total_paras) * 100,
                'unlabeled_percentage': (total_paragraph_actions['unlabeled_paragraphs'] / total_paras) * 100,
                'campaigns_analyzed': len(paragraph_campaigns)
            }
    
    return aggregated

def calculate_player_campaign_participation(all_results: Dict) -> Dict[str, int]:

    """
    Calculate the number of campaigns each player participated in.
    
    Args:
        all_results (Dict): Dictionary with campaign analysis results from analyze_all_campaigns()
                           Expected structure: {'per_campaign': {campaign_id: {...}}}
        
    Returns:
        Dict[str, int]: Dictionary with player names as keys and campaign count as values
                       Sorted by campaign count (descending) then by player name
        
    Example:
        {'alice': 5, 'bob': 3, 'charlie': 2, 'david': 1}
    """
    player_campaign_counts = {}
    
    # Extract per-campaign results
    per_campaign_results = all_results.get('per_campaign', {})
    
    if not per_campaign_results:
        print("Warning: No per-campaign results found in all_results")
        return {}
    
    # Iterate through each campaign's results
    for campaign_id, campaign_results in per_campaign_results.items():
        # Get players from the post_lengths_by_player analysis
        post_lengths_by_player = campaign_results.get('post_lengths_by_player', {})
        
        # Extract unique players from this campaign
        campaign_players = set()
        
        # Add players from post_lengths_by_player results
        for player_name in post_lengths_by_player.keys():
            if player_name != 'overall':  # Skip overall statistics
                campaign_players.add(player_name)
        
        # Also check time_intervals_by_player as backup
        time_intervals_by_player = campaign_results.get('time_intervals_by_player', {})
        for player_name in time_intervals_by_player.keys():
            if player_name != 'overall':
                campaign_players.add(player_name)
        
        # Also check unique_players_characters for completeness
        unique_players_chars = campaign_results.get('unique_players_characters', {})
        final_players = unique_players_chars.get('final_unique_players', 0)
        
        # If we have players from the analysis results, count them
        for player_name in campaign_players:
            if player_name in player_campaign_counts:
                player_campaign_counts[player_name] += 1
            else:
                player_campaign_counts[player_name] = 1
    
    # Sort by campaign count (descending), then by player name (ascending)
    sorted_players = sorted(player_campaign_counts.items(), 
                           key=lambda x: (-x[1], x[0]))
    
    return dict(sorted_players)

# ===================================================================
# Unified CACHING SYSTEM
# ===================================================================

### Save ###
def save_campaign_results(all_results: Dict, num_campaigns: int, 
                         data_file_path: Optional[str] = None,
                         cache_dir: Optional[str] = None) -> str:
    """
    Save campaign analysis results to cache with metadata.
    
    Args:
        all_results: Complete results from analyze_all_campaigns()
        num_campaigns: Number of campaigns analyzed
        data_file_path: Path to source data file for validation. If None, uses default repo path.
        cache_dir: Directory to store cache files. If None, uses default repo path.
        
    Returns:
        str: Path to saved cache file
    """
    # Set default paths relative to repository root
    repo_root = Path(__file__).parent.parent
    if data_file_path is None:
        data_file_path = str(repo_root / 'data' / 'raw-human-games' / 'individual_campaigns')
    if cache_dir is None:
        cache_dir = str(repo_root / 'data' / 'processed' / 'campaign_stats_cache')
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename
    cache_filename = f"campaign_analysis_{num_campaigns}_campaigns.pkl"
    cache_file_path = cache_path / cache_filename
    
    # Prepare metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_campaigns': num_campaigns,
        'data_file_path': data_file_path,
        'data_file_hash': _get_data_file_hash(data_file_path),
        'total_messages': all_results.get('summary_stats', {}).get('total_messages', 0),
        'cache_version': '1.0'
    }
    
    # Combine results with metadata
    cache_data = {
        'metadata': metadata,
        'results': all_results
    }
    
    # Save to file
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Print confirmation
    file_size = os.path.getsize(cache_file_path) / (1024 * 1024)  # MB
    print(f"💾 Saved cache: {cache_filename} ({file_size:.2f} MB)")
    
    return str(cache_file_path)

def save_creativity_results(creativity_results: Dict, num_campaigns: int, cache_dir: Optional[str] = None) -> str:
    """
    Save creativity analysis results to cache.
    
    Parameters
    ----------
    creativity_results : Dict
        Results from analyze_creativity_all_campaigns()
    num_campaigns : int
        Number of campaigns analyzed
    cache_dir : str, optional
        Directory to store cached results. If None, uses default repo path.
        
    Returns
    -------
    str
        Path to saved cache file
    """
    if cache_dir is None:
        repo_root = Path(__file__).parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'creativity_cache')
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'creativity_analysis_{num_campaigns}_campaigns.pkl')
    
    with open(cache_file, 'wb') as f:
        pickle.dump(creativity_results, f)
    
    return cache_file

### Load ###
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
    cache_file = os.path.join(cache_dir, f'creativity_analysis_{num_campaigns}_campaigns.pkl')
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    return None

def load_campaign_results(num_campaigns: int, 
                         data_file_path: str = 'Game-Data/data-labels.json',
                         cache_dir: str = 'campaign_stats_cache',
                         validate_freshness: bool = True) -> Optional[Dict]:
    """
    Load cached campaign analysis results.
    
    Args:
        num_campaigns: Number of campaigns to load
        data_file_path: Path to source data file for validation
        cache_dir: Directory containing cache files
        validate_freshness: Whether to validate against source data
        
    Returns:
        Dict with cached results or None if not found/invalid
    """
    cache_path = Path(cache_dir)
    cache_filename = f"campaign_analysis_{num_campaigns}_campaigns.pkl"
    cache_file_path = cache_path / cache_filename
    
    # Check if cache file exists
    if not cache_file_path.exists():
        return None
    
    try:
        # Load cache data
        with open(cache_file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Extract metadata and results
        metadata = cache_data.get('metadata', {})
        results = cache_data.get('results', {})
        
        # Validate cache if requested
        if validate_freshness:
            # Check if source data file has changed
            current_hash = _get_data_file_hash(data_file_path)
            cached_hash = metadata.get('data_file_hash', '')
            
            if current_hash != cached_hash:
                print(f"⚠️  Cache invalid: source data has changed since cache creation")
                return None
        
        # Print cache info
        cache_date = metadata.get('timestamp', 'Unknown')
        file_size = os.path.getsize(cache_file_path) / (1024 * 1024)  # MB
        print(f"📁 Loaded cache: {cache_filename} (created: {cache_date[:19]}, {file_size:.2f} MB)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading cache {cache_filename}: {e}")
        return None

### load or compute ###
def load_or_compute_incremental(max_campaigns: int,
                               data_file_path: str = 'Game-Data/data-labels.json',
                               cache_dir: str = 'campaign_stats_cache',
                               force_refresh: bool = False,
                               show_progress: bool = True) -> Dict:
    """
    Intelligently load cached results or compute incrementally.
    
    Args:
        max_campaigns: Target number of campaigns to analyze
        data_file_path: Path to campaign data file
        cache_dir: Directory for cache files
        force_refresh: If True, bypass all caching and recompute
        show_progress: Whether to show progress indicators
        
    Returns:
        Dict: Complete analysis results for requested number of campaigns
    """
    if show_progress:
        print(f"🎯 Target: Analysis of {max_campaigns} campaigns")
    
    # Force refresh - skip all caching
    if force_refresh:
        if show_progress:
            print("🔄 Force refresh requested - running fresh analysis...")
        
        # Load campaigns and run analysis
        campaign_dfs, json_data = dl.load_campaigns(data_file_path, max_campaigns, show_progress, return_json=True)
        if not campaign_dfs:
            raise ValueError("Failed to load campaigns")
        
        results = analyze_all_campaigns(campaign_dfs, json_data, show_progress)
        
        # Save new cache
        save_campaign_results(results, max_campaigns, data_file_path, cache_dir)
        return results
    
    # Check for exact cache match
    exact_results = load_campaign_results(max_campaigns, data_file_path, cache_dir)
    if exact_results is not None:
        if show_progress:
            print(f"✅ Found exact cached results for {max_campaigns} campaigns")
        return exact_results
    
    # Find largest available cache smaller than target
    available_caches = get_available_cached_results(cache_dir)
    suitable_caches = [cache for cache in available_caches if cache['num_campaigns'] < max_campaigns]
    
    if not suitable_caches:
        # No suitable cache found - run fresh analysis
        if show_progress:
            print("📊 No cached results found - running fresh analysis...")
        
        campaign_dfs, json_data = dl.load_campaigns(data_file_path, max_campaigns, show_progress, return_json=True)
        if not campaign_dfs:
            raise ValueError("Failed to load campaigns")
        
        results = analyze_all_campaigns(campaign_dfs, json_data, show_progress)
        save_campaign_results(results, max_campaigns, data_file_path, cache_dir)
        return results
    
    # Use largest suitable cache for incremental computation
    largest_cache = suitable_caches[-1]  # Already sorted by campaign count
    cached_campaigns = largest_cache['num_campaigns']
    
    if show_progress:
        print(f"📁 Found cached results for {cached_campaigns} campaigns")
        print(f"🔄 Computing additional {max_campaigns - cached_campaigns} campaigns ({cached_campaigns + 1}-{max_campaigns})...")
    
    # Load cached results
    cached_results = load_campaign_results(cached_campaigns, data_file_path, cache_dir)
    if cached_results is None:
        raise ValueError(f"Failed to load cached results for {cached_campaigns} campaigns")
    
    # Load additional campaigns
    all_campaign_dfs, all_json_data = dl.load_campaigns(data_file_path, max_campaigns, show_progress=False, return_json=True)
    if not all_campaign_dfs:
        raise ValueError("Failed to load campaigns")
    
    # Get campaign IDs from cache to determine which are new
    cached_campaign_ids = set(cached_results['per_campaign'].keys())
    all_campaign_ids = list(all_campaign_dfs.keys())
    
    # Determine new campaigns to process (those beyond cached count)
    new_campaign_ids = all_campaign_ids[cached_campaigns:max_campaigns]
    new_campaign_dfs = {cid: all_campaign_dfs[cid] for cid in new_campaign_ids}
    new_json_data = {cid: all_json_data[cid] for cid in new_campaign_ids}
    
    if show_progress and new_campaign_dfs:
        print(f"⚡ Processing {len(new_campaign_dfs)} additional campaigns...")
    
    # Analyze new campaigns
    if new_campaign_dfs:
        new_results = analyze_all_campaigns(new_campaign_dfs, new_json_data, show_progress)
        
        # Merge results
        if show_progress:
            print("🔄 Merging cached and new results...")
        merged_results = _merge_campaign_results(cached_results, new_results)
    else:
        # No new campaigns to process
        merged_results = cached_results
    
    # Save combined results
    save_campaign_results(merged_results, max_campaigns, data_file_path, cache_dir)
    
    if show_progress:
        print(f"✅ Incremental analysis complete: {max_campaigns} campaigns total")
    
    return merged_results

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
    
    campaign_iterator = tqdm(campaigns_to_compute, desc="Computing additional creativity metrics") if show_progress else campaigns_to_compute
    
    for campaign_id in campaign_iterator:
        try:
            # Load single campaign data
            campaign_data = {campaign_id: all_data[campaign_id]}
            df = dl._load_dnd_data(campaign_data)
            
            if len(df) == 0:
                continue
            
            campaign_results = {}
            
            # Get embeddings for all text (combined)
            embeddings = creativity.get_embeddings(df)
            campaign_results['embeddings'] = embeddings
            
            # Get label-aware embeddings
            try:
                label_embeddings = creativity.get_embeddings_by_label(df)
                campaign_results['label_embeddings'] = label_embeddings
            except Exception as e:
                if show_progress:
                    print(f"Label-aware embeddings failed for campaign {campaign_id}: {e}")
                campaign_results['label_embeddings'] = {}
            
            # Calculate semantic distances (combined)
            semantic_distances = creativity.semantic_distance(df, embeddings=embeddings)
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
                            label_distances = creativity.semantic_distance(label_df, embeddings=label_emb)
                            label_semantic_distances[label] = label_distances
                        except Exception as e:
                            if show_progress:
                                print(f"Label distance calculation failed for {label}: {e}")
            
            campaign_results['label_semantic_distances'] = label_semantic_distances
            
            # Analyze session novelty (combined)
            novelty_results = creativity.session_novelty(df, embeddings=embeddings)
            campaign_results['session_novelty'] = novelty_results
            
            # Topic modeling
            try:
                topics_series, topic_model_obj = creativity.topic_model(df, embeddings=embeddings)
                campaign_results['topic_model'] = {
                    'topics': topics_series,
                    'model': topic_model_obj
                }
                
                # Add topics to DataFrame for transition analysis
                df_with_topics = df.copy()
                df_with_topics['topic'] = topics_series
                
                # Topic transitions
                topic_transitions = creativity.topic_transition_matrix(df_with_topics, topic_col='topic')
                campaign_results['topic_transitions'] = topic_transitions
                
                # Topic change rate
                change_rate = creativity.topic_change_rate(df_with_topics, topic_col='topic')
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

### get info ###
def get_available_cached_results(cache_dir: str = 'campaign_stats_cache') -> List[Dict]:
    """
    List all available cached analysis results.
    
    Args:
        cache_dir: Directory containing cache files
        
    Returns:
        List of dicts with cache information, sorted by campaign count
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return []
    
    cache_info = []
    
    # Find all cache files
    for cache_file in cache_path.glob("campaign_analysis_*_campaigns.pkl"):
        try:
            # Extract campaign count from filename
            filename = cache_file.name
            num_campaigns = int(filename.split('_')[2])
            
            # Get file info
            stat = cache_file.stat()
            file_size_mb = stat.st_size / (1024 * 1024)
            modification_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Try to load metadata
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                metadata = cache_data.get('metadata', {})
                total_messages = metadata.get('total_messages', 'Unknown')
                cache_timestamp = metadata.get('timestamp', modification_time.isoformat())
            except:
                total_messages = 'Unknown'
                cache_timestamp = modification_time.isoformat()
            
            cache_info.append({
                'num_campaigns': num_campaigns,
                'filename': filename,
                'file_size_mb': file_size_mb,
                'creation_date': cache_timestamp[:19],
                'total_messages': total_messages,
                'file_path': str(cache_file)
            })
            
        except (ValueError, IndexError):
            # Skip files that don't match expected pattern
            continue
    
    # Sort by campaign count
    cache_info.sort(key=lambda x: x['num_campaigns'])
    
    return cache_info

def show_cache_status(cache_dir: str = 'campaign_stats_cache') -> None:
    """
    Display detailed information about cached results.
    
    Args:
        cache_dir: Directory containing cache files
    """
    available_caches = get_available_cached_results(cache_dir)
    
    if not available_caches:
        print(f"No cached results found in {cache_dir}")
        return
    
    print(f"📊 CACHED ANALYSIS RESULTS ({len(available_caches)} files)")
    print(f"Cache directory: {cache_dir}")
    print()
    
    total_size = 0
    
    for cache in available_caches:
        total_size += cache['file_size_mb']
        messages_str = f"{cache['total_messages']:,}" if isinstance(cache['total_messages'], int) else str(cache['total_messages'])
        
        print(f"  📁 {cache['filename']}")
        print(f"     Campaigns: {cache['num_campaigns']}")
        print(f"     Messages: {messages_str}")
        print(f"     Created: {cache['creation_date']}")
        print(f"     Size: {cache['file_size_mb']:.2f} MB")
        print()
    
    print(f"Total cache size: {total_size:.2f} MB")

### clear ###
def clear_cache(cache_dir: str = 'campaign_stats_cache', 
               confirm: bool = True) -> None:
    """
    Clear all cached analysis results.
    
    Args:
        cache_dir: Directory containing cache files
        confirm: Whether to ask for confirmation
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist")
        return
    
    cache_files = list(cache_path.glob("campaign_analysis_*.pkl"))
    
    if not cache_files:
        print(f"No cache files found in {cache_dir}")
        return
    
    if confirm:
        response = input(f"Delete {len(cache_files)} cache files? (y/N): ")
        if response.lower() != 'y':
            print("Cache clearing cancelled")
            return
    
    deleted_count = 0
    total_size = 0
    
    for cache_file in cache_files:
        try:
            total_size += cache_file.stat().st_size
            cache_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {cache_file.name}: {e}")
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"🗑️  Deleted {deleted_count} cache files ({total_size_mb:.2f} MB freed)")

### helpers ###
def _get_data_file_hash(data_file_path: str) -> str:
    """Calculate hash of data file for cache validation."""
    if not os.path.exists(data_file_path):
        return ""
    
    # Use file size and modification time for quick validation
    stat = os.stat(data_file_path)
    hash_string = f"{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_string.encode()).hexdigest()

def _merge_campaign_results(cached_results: Dict, new_results: Dict) -> Dict:
    """
    Merge cached campaign results with newly computed results.
    
    Args:
        cached_results: Previously cached analysis results
        new_results: Newly computed analysis results
        
    Returns:
        Dict: Combined results
    """
    # Create merged results structure
    merged_results = {
        'per_campaign': {},
        'aggregated': {},
        'summary_stats': {}
    }
    
    # Merge per-campaign results
    merged_results['per_campaign'].update(cached_results.get('per_campaign', {}))
    merged_results['per_campaign'].update(new_results.get('per_campaign', {}))
    
    # Re-aggregate all results (easier than merging aggregated stats)
    print("🔄 Re-aggregating combined results...")
    merged_results['aggregated'] = aggregate_campaign_metrics(merged_results['per_campaign'])
    
    # Generate new summary stats
    # We need to reconstruct campaign_dataframes for summary stats
    # For now, we'll combine the basic stats from both caches
    cached_summary = cached_results.get('summary_stats', {})
    new_summary = new_results.get('summary_stats', {})
    
    merged_results['summary_stats'] = {
        'total_campaigns': cached_summary.get('total_campaigns', 0) + new_summary.get('total_campaigns', 0),
        'total_messages': cached_summary.get('total_messages', 0) + new_summary.get('total_messages', 0),
        'campaign_size_stats': new_summary.get('campaign_size_stats', {}),  # Use new stats
        'duration_stats': new_summary.get('duration_stats', {}),
        'player_count_stats': new_summary.get('player_count_stats', {}),
        'activity_stats': new_summary.get('activity_stats', {})
    }
    
    return merged_results
