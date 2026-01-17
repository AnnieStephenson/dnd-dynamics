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
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import other modules using relative imports
from . import data_loading as dl
from . import basic_metrics as basic
from . import creativity_metrics as creativity


# ===================================================================
# Aggregation functions
# ===================================================================

def aggregate_creativity_metrics(all_creativity_results: Dict) -> Dict:

    """
    Combine creativity results from multiple campaigns for statistical analysis.
    
    Parameters
    ----------
    all_creativity_results : Dict
        Output from analyze_creativity()
        
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
# Per-Campaign Caching System
# ===================================================================

def load_cached_results(cache_dir: str, campaign_ids: List[str]) -> Dict[str, Any]:
    """
    Load cached results for specified campaigns.
    
    Args:
        cache_dir: Directory containing cache files
        campaign_ids: List of campaign IDs to load
        
    Returns:
        Dict mapping campaign_id to cached results (only for campaigns that have cache files)
    """
    cache_path = Path(cache_dir)
    cached_results = {}
    
    if not cache_path.exists():
        return cached_results
    
    for campaign_id in campaign_ids:
        cache_file = cache_path / f"{campaign_id}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_results[campaign_id] = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load cache for {campaign_id}: {e}")
    
    return cached_results


def save_cached_results(cache_dir: str, results: Dict[str, Any]) -> None:
    """
    Save results to individual campaign cache files.
    
    Args:
        cache_dir: Directory to store cache files
        results: Dict mapping campaign_id to results
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    for campaign_id, result in results.items():
        cache_file = cache_path / f"{campaign_id}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save cache for {campaign_id}: {e}")


def get_missing_campaigns(cache_dir: str, campaign_ids: List[str]) -> List[str]:
    """
    Return campaign IDs that don't have cached results.
    
    Args:
        cache_dir: Directory containing cache files
        campaign_ids: List of campaign IDs to check
        
    Returns:
        List of campaign IDs that need computation
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return campaign_ids
    
    missing = []
    for campaign_id in campaign_ids:
        cache_file = cache_path / f"{campaign_id}.pkl"
        if not cache_file.exists():
            missing.append(campaign_id)
    
    return missing


def get_cache_status(cache_dir: str, campaign_ids: List[str]) -> Dict[str, bool]:
    """
    Get cache status for a list of campaigns.
    
    Args:
        cache_dir: Directory containing cache files
        campaign_ids: List of campaign IDs to check
        
    Returns:
        Dict mapping campaign_id to cache status (True if cached, False if not)
    """
    cache_path = Path(cache_dir)
    status = {}
    
    for campaign_id in campaign_ids:
        if cache_path.exists():
            cache_file = cache_path / f"{campaign_id}.pkl"
            status[campaign_id] = cache_file.exists()
        else:
            status[campaign_id] = False
    
    return status


def handle_multi_campaign_caching(data: Dict[str, pd.DataFrame],
    cache_dir: str,
    force_refresh: bool,
    show_progress: bool,
    analysis_name: str
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
        
    Returns:
        Tuple of (cached_results, data_to_process) where:
        - cached_results: Dict of previously cached results {campaign_id: results}
        - data_to_process: Dict of DataFrames that need analysis {campaign_id: df}
    """
    campaign_ids = list(data.keys())
    
    # Handle caching
    if not force_refresh:
        # Load cached results
        cached_results = load_cached_results(cache_dir, campaign_ids)
        missing_campaigns = get_missing_campaigns(cache_dir, campaign_ids)
        
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
    analysis_name: str
) -> Dict[str, Any]:
    """
    Save new analysis results to cache and combine with cached results.
    
    Args:
        cached_results: Previously cached results
        new_results: Newly computed results
        cache_dir: Directory for caching results
        show_progress: Whether to show progress indicators
        analysis_name: Name of analysis for progress messages
        
    Returns:
        Combined results dictionary
    """
    # Save new results to cache
    if new_results:
        save_cached_results(cache_dir, new_results)
        if show_progress:
            print(f"💾 Saved {len(new_results)} {analysis_name} results to cache")
    
    # Combine cached and new results
    results = {**cached_results, **new_results}
    return results
