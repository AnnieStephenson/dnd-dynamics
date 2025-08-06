"""
Basic D&D Campaign Analysis Metrics

This module provides fundamental analysis functions for D&D gameplay logs including 
time patterns, posting behavior, message characteristics, and player activity metrics.
Functions calculate statistics on time intervals, post lengths, character mentions, 
dice rolls, and action classifications to understand campaign dynamics and player engagement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from . import data_loading as dl
from . import batch
from tqdm import tqdm

# ===================================================================
# User facing multi-parameter and/or multi campaign
# ===================================================================

def _analyze_single_campaign_basic_metrics(df: pd.DataFrame) -> Dict:
    """
    Run all basic analysis functions for a single campaign.
    
    Args:
        df: Campaign DataFrame
        
    Returns:
        Dict with basic analysis results
    """
    return {
        'time_intervals_overall': analyze_time_intervals(df, by_player=False),
        'time_intervals_by_player': analyze_time_intervals(df, by_player=True),
        'cumulative_posts_overall': analyze_cumulative_posts(df, by_player=False),
        'cumulative_posts_by_player': analyze_cumulative_posts(df, by_player=True),
        'unique_players_characters': analyze_unique_players_characters(df),
        'post_lengths_overall': analyze_post_lengths(df, by_player=False),
        'post_lengths_by_player': analyze_post_lengths(df, by_player=True),
        'post_lengths_by_label_overall': analyze_post_lengths_by_label(df, by_player=False),
        'post_lengths_by_label_by_player': analyze_post_lengths_by_label(df, by_player=True),
        'character_mentions': analyze_character_mentions(df),
        'dice_roll_frequency': analyze_dice_roll_frequency(df),
        'summary_report': generate_summary_report(df)
    }


def analyze_basic_metrics(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Union[Dict, Dict[str, Dict]]:
    """
    Analyze basic metrics for single or multiple campaigns using DataFrames.
    
    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/basic_results)
        force_refresh: Whether to force recomputation even if cached results exist
        
    Returns:
        Dict of basic metrics for single campaign, or Dict[campaign_id, metrics] for multiple
    """
    if isinstance(data, pd.DataFrame):
        # Single campaign analysis - no caching for single campaigns
        return _analyze_single_campaign_basic_metrics(data)
    
    elif isinstance(data, dict):
        # Multiple campaign analysis with caching support
        
        # Set default cache directory
        if cache_dir is None:
            repo_root = Path(__file__).parent.parent
            cache_dir = str(repo_root / 'data' / 'processed' / 'basic_results')
        
        # Handle caching using helper function
        cached_results, data_to_process = batch.handle_multi_campaign_caching(
            data, cache_dir, force_refresh, show_progress, "basic metrics"
        )
        
        # Process missing campaigns
        new_results = {}
        if data_to_process:
            if show_progress and len(data_to_process) > 1:
                iterator = tqdm(data_to_process.items(), desc="Analyzing campaigns", total=len(data_to_process))
            else:
                iterator = data_to_process.items()
            
            for campaign_id, df in iterator:
                new_results[campaign_id] = _analyze_single_campaign_basic_metrics(df)
        
        # Save new results and combine with cached results
        return batch.save_new_results_and_combine(
            cached_results, new_results, cache_dir, show_progress, "basic metrics"
        )
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.DataFrame or Dict[str, pd.DataFrame]")


def list_campaigns_by_size(campaign_dataframes: Dict[str, pd.DataFrame],
                          top_n: int = 10) -> List[Tuple[str, int]]:
    """
    List campaigns sorted by size (number of messages).
    
    Args:
        campaign_dataframes: Dictionary of campaign DataFrames
        top_n: Number of top campaigns to return
        
    Returns:
        List of (campaign_id, message_count) tuples sorted by size
    """
    campaign_sizes = [(cid, len(df)) for cid, df in campaign_dataframes.items()]
    campaign_sizes.sort(key=lambda x: x[1], reverse=True)
    return campaign_sizes[:top_n]


def get_campaign_sample(campaign_dataframes: Dict[str, pd.DataFrame], 
                       campaign_id: str) -> Optional[pd.DataFrame]:
    """
    Get a specific campaign's DataFrame for individual analysis.
    
    Args:
        campaign_dataframes: Dictionary of campaign DataFrames
        campaign_id: ID of the campaign to retrieve
        
    Returns:
        DataFrame for the specified campaign, or None if not found
    """
    return campaign_dataframes.get(campaign_id)


# ===================================================================
# SINGLE DF FUNCTIONS
# ===================================================================


def analyze_time_intervals(df: pd.DataFrame, by_player: bool = False) -> Dict:
    """
    Analyze time intervals between consecutive posts.
    
    Args:
        df: DataFrame containing D&D messages
        by_player: If True, analyze intervals per player
        
    Returns:
        Dict containing interval statistics and histogram data for plotting
    """
    results = {}

    if by_player and 'player' in df.columns:
        players = df['player'].dropna().unique()  # Include all players

        for player in players:
            player_df = df[df['player'] == player].sort_values('date')
            intervals = player_df['date'].diff().dt.total_seconds() / 3600  # Convert to hours
            intervals = intervals.dropna()

            if len(intervals) > 0:
                results[player] = {
                    'mean_hours': intervals.mean(),
                    'median_hours': intervals.median(),
                    'std_hours': intervals.std(),
                    'count': len(intervals),
                    'intervals_data': intervals.values  # For histogram
                }
    else:
        # Overall analysis
        df_sorted = df.sort_values('date')
        intervals = df_sorted['date'].diff().dt.total_seconds() / 3600  # Convert to hours
        intervals = intervals.dropna()

        results['overall'] = {
            'mean_hours': intervals.mean(),
            'median_hours': intervals.median(),
            'std_hours': intervals.std(),
            'count': len(intervals),
            'intervals_data': intervals.values
        }

    return results


def analyze_cumulative_posts(df: pd.DataFrame, by_player: bool = False) -> Dict:
    """
    Analyze cumulative post count over time.
    
    Args:
        df: DataFrame containing D&D messages
        by_player: If True, return cumulative posts per player
        
    Returns:
        Dict containing cumulative post data for plotting
    """
    df_sorted = df.sort_values('date').copy()

    if by_player and 'player' in df.columns:
        # Create cumulative counts by player
        player_counts = df_sorted.groupby(['date', 'player']).size().unstack(fill_value=0)
        cumulative_by_player = player_counts.cumsum()

        # Filter to active players only
        active_players = [col for col in cumulative_by_player.columns
                         if cumulative_by_player[col].max() > 5]

        return {
            'type': 'by_player',
            'data': cumulative_by_player[active_players],
            'dates': cumulative_by_player.index,
            'players': active_players
        }
    else:
        # Overall cumulative count
        df_sorted['cumulative_posts'] = range(1, len(df_sorted) + 1)

        return {
            'type': 'overall',
            'dates': df_sorted['date'].values,
            'cumulative_posts': df_sorted['cumulative_posts'].values
        }


def analyze_unique_players_characters(df: pd.DataFrame) -> Dict:
    """
    Analyze number of unique players and characters over time.
    
    Args:
        df: DataFrame containing D&D messages
        
    Returns:
        Dict containing unique counts data for plotting
    """
    df_sorted = df.sort_values('date').copy()

    # Calculate cumulative unique players and characters
    unique_players = []
    unique_characters = []
    seen_players = set()
    seen_characters = set()

    for _, row in df_sorted.iterrows():
        if pd.notna(row['player']):
            seen_players.add(row['player'])
        if pd.notna(row['character']):
            seen_characters.add(row['character'])

        unique_players.append(len(seen_players))
        unique_characters.append(len(seen_characters))

    return {
        'final_unique_players': len(seen_players),
        'final_unique_characters': len(seen_characters),
        'dates': df_sorted['date'].values,
        'unique_players_cumulative': unique_players,
        'unique_characters_cumulative': unique_characters
    }


def analyze_post_lengths(df: pd.DataFrame, by_player: bool = False) -> Dict:
    """
    Analyze histogram of post lengths (word count).
    
    Args:
        df: DataFrame containing D&D messages
        by_player: If True, return distribution per player
        
    Returns:
        Dict containing length statistics and histogram data
    """
    results = {}

    if by_player and 'player' in df.columns:
        # Analysis by player - include all players
        all_players = df['player'].dropna().unique()

        for player in all_players:
            player_df = df[df['player'] == player]
            word_counts = player_df['word_count']

            results[player] = {
                'mean_words': word_counts.mean(),
                'median_words': word_counts.median(),
                'std_words': word_counts.std(),
                'max_words': word_counts.max(),
                'count': len(word_counts),
                'word_counts_data': word_counts.values  # For histogram
            }
    else:
        # Overall analysis
        word_counts = df['word_count']

        results['overall'] = {
            'mean_words': word_counts.mean(),
            'median_words': word_counts.median(),
            'std_words': word_counts.std(),
            'max_words': word_counts.max(),
            'count': len(word_counts),
            'word_counts_data': word_counts.values  # For histogram
        }

    return results


def analyze_post_lengths_by_label(df: pd.DataFrame, by_player: bool = False) -> Dict:
    """
    Analyze post lengths separated by character label (in-character, out-of-character, mixed).
    
    Args:
        df: DataFrame containing D&D messages with label information
        by_player: If True, return distribution per player
        
    Returns:
        Dict containing length statistics separated by label type
    """
    results = {
        'in_character': {},
        'out_of_character': {},
        'mixed': {},
        'unlabeled': {}
    }

    # Define label columns mapping
    label_columns = {
        'in_character': 'in_character_word_count',
        'out_of_character': 'out_of_character_word_count',
        'mixed': 'mixed_word_count'
    }

    if by_player and 'player' in df.columns:
        # Analysis by player for each label type
        all_players = df['player'].dropna().unique()

        for label_type, word_col in label_columns.items():
            results[label_type] = {}

            for player in all_players:
                player_df = df[df['player'] == player]
                word_counts = player_df[word_col]
                # Only include messages that have content for this label
                word_counts = word_counts[word_counts > 0]

                if len(word_counts) > 0:
                    results[label_type][player] = {
                        'mean_words': word_counts.mean(),
                        'median_words': word_counts.median(),
                        'std_words': word_counts.std(),
                        'max_words': word_counts.max(),
                        'count': len(word_counts),
                        'total_words': word_counts.sum(),
                        'word_counts_data': word_counts.values
                    }

        # Handle unlabeled content (uses primary_label)
        results['unlabeled'] = {}
        for player in all_players:
            player_df = df[(df['player'] == player) & (df['primary_label'] == 'unlabeled')]
            if len(player_df) > 0:
                word_counts = player_df['word_count']
                word_counts = word_counts[word_counts > 0]

                if len(word_counts) > 0:
                    results['unlabeled'][player] = {
                        'mean_words': word_counts.mean(),
                        'median_words': word_counts.median(),
                        'std_words': word_counts.std(),
                        'max_words': word_counts.max(),
                        'count': len(word_counts),
                        'total_words': word_counts.sum(),
                        'word_counts_data': word_counts.values
                    }
    else:
        # Overall analysis for each label type
        for label_type, word_col in label_columns.items():
            word_counts = df[word_col]
            word_counts = word_counts[word_counts > 0]  # Only non-empty content

            if len(word_counts) > 0:
                results[label_type]['overall'] = {
                    'mean_words': word_counts.mean(),
                    'median_words': word_counts.median(),
                    'std_words': word_counts.std(),
                    'max_words': word_counts.max(),
                    'count': len(word_counts),
                    'total_words': word_counts.sum(),
                    'word_counts_data': word_counts.values
                }

        # Handle unlabeled content
        unlabeled_df = df[df['primary_label'] == 'unlabeled']
        if len(unlabeled_df) > 0:
            word_counts = unlabeled_df['word_count']
            word_counts = word_counts[word_counts > 0]

            if len(word_counts) > 0:
                results['unlabeled']['overall'] = {
                    'mean_words': word_counts.mean(),
                    'median_words': word_counts.median(),
                    'std_words': word_counts.std(),
                    'max_words': word_counts.max(),
                    'count': len(word_counts),
                    'total_words': word_counts.sum(),
                    'word_counts_data': word_counts.values
                }

    # Add summary statistics
    results['summary'] = {
        'total_messages': len(df),
        'messages_with_in_character': len(df[df['in_character_word_count'] > 0]),
        'messages_with_out_of_character': len(df[df['out_of_character_word_count'] > 0]),
        'messages_with_mixed': len(df[df['mixed_word_count'] > 0]),
        'messages_unlabeled': len(df[df['primary_label'] == 'unlabeled']),
        'avg_labels_per_message': df['label_counts'].apply(lambda x: sum(x.values())).mean()
    }

    return results


def analyze_character_mentions(df: pd.DataFrame, top_n: int = 15) -> Dict:
    """
    Analyze most frequently mentioned character names.
    
    Args:
        df: DataFrame containing D&D messages
        top_n: Number of top characters to include
        
    Returns:
        Dict containing character mention statistics and data for plotting
    """
    # Flatten all name mentions
    all_mentions = []
    for mentions in df['name_mentions'].dropna():
        if isinstance(mentions, list):
            all_mentions.extend(mentions)

    # Count mentions
    mention_counts = pd.Series(all_mentions).value_counts()
    top_mentions = mention_counts.head(top_n)

    return {
        'total_mentions': len(all_mentions),
        'unique_characters_mentioned': len(mention_counts),
        'top_mentions': top_mentions.to_dict(),
        'top_mentions_names': top_mentions.index.tolist(),
        'top_mentions_counts': top_mentions.values.tolist(),
        'full_counts': mention_counts.to_dict()
    }


def analyze_dice_roll_frequency(df: pd.DataFrame) -> Dict:
    """
    Analyze fraction of posts containing dice rolls.
    
    Args:
        df: DataFrame containing D&D messages
        
    Returns:
        Dict containing dice roll statistics and data for plotting
    """
    total_posts = len(df)
    posts_with_rolls = df['has_dice_roll'].sum()
    roll_percentage = (posts_with_rolls / total_posts) * 100

    # Analyze by player
    player_roll_stats = df.groupby('player')['has_dice_roll'].agg(['count', 'sum']).reset_index()
    player_roll_stats['percentage'] = (player_roll_stats['sum'] / player_roll_stats['count']) * 100
    player_roll_stats = player_roll_stats[player_roll_stats['count'] >= 5]  # Filter players with <5 posts
    player_roll_stats = player_roll_stats.sort_values('percentage', ascending=False)

    # Time series of dice rolls
    df_sorted = df.sort_values('date').copy()
    daily_rolls = df_sorted.groupby(df_sorted['date'].dt.date).agg({
        'has_dice_roll': ['count', 'sum']
    }).reset_index()
    daily_rolls.columns = ['date', 'total_posts', 'posts_with_rolls']
    daily_rolls['roll_percentage'] = (daily_rolls['posts_with_rolls'] / daily_rolls['total_posts']) * 100

    return {
        'total_posts': total_posts,
        'posts_with_rolls': posts_with_rolls,
        'roll_percentage': roll_percentage,
        'posts_without_rolls': total_posts - posts_with_rolls,
        'player_stats': player_roll_stats.to_dict('records'),
        'player_names': player_roll_stats['player'].tolist(),
        'player_percentages': player_roll_stats['percentage'].tolist(),
        'daily_data': daily_rolls,
        'daily_dates': daily_rolls['date'].tolist(),
        'daily_percentages': daily_rolls['roll_percentage'].tolist()
    }


def generate_summary_report(df: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive summary report of the D&D campaign.
    
    Args:
        df: DataFrame containing D&D messages
        
    Returns:
        Dict containing comprehensive campaign statistics
    """
    total_posts = len(df)
    date_range = df['date'].max() - df['date'].min()
    unique_players = df['player'].nunique()
    unique_characters = df['character'].nunique()

    report = {
        'campaign_overview': {
            'total_posts':
            total_posts,
            'campaign_duration_days':
            date_range.days,
            'unique_players':
            unique_players,
            'unique_characters':
            unique_characters,
            'posts_per_day':
            total_posts / max(date_range.days, 1),
            'date_range':
            f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        },
        'posting_patterns': {
            'most_active_player':
            df['player'].value_counts().index[0] if len(df) > 0 else None,
            'posts_by_most_active':
            df['player'].value_counts().iloc[0] if len(df) > 0 else 0,
            'average_post_length':
            df['word_count'].mean(),
            'longest_post_words':
            df['word_count'].max()
        },
        'gameplay_characteristics': {
            'dice_roll_percentage':
            (df['has_dice_roll'].sum() / total_posts) * 100,
            'action_percentage':
            (df['message_type'].value_counts().get('action', 0) / total_posts)
            * 100,
            'dialogue_percentage':
            (df['message_type'].value_counts().get('dialogue', 0) /
             total_posts) * 100,
            'combat_posts':
            df['in_combat'].sum(),
            'combat_percentage': (df['in_combat'].sum() / total_posts) * 100
        }
    }

    return report


# ===================================================================
# JSON FUNCTIONS
# ===================================================================

def analyze_paragraph_actions(json_data: Dict) -> Dict:
    """
    Analyze paragraph-level action types based on actual data structure.
    
    This function examines the 'actions' field in each paragraph to count
    different types of actions: name_mentions, spells, dialogue, roll, weapon.
    
    Args:
        json_data: Dictionary containing campaign data with paragraph structure
        
    Returns:
        Dict containing paragraph-level action statistics and character label counts
    """

    # Initialize counters
    action_counts = {
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

    # Track daily trends for time series analysis
    daily_action_counts = {}

    # Process each campaign and message
    for campaign_id, messages in json_data.items():
        for message_id, message_data in messages.items():

            # Process paragraphs if they exist
            if 'paragraphs' in message_data:
                paragraphs = message_data.get('paragraphs', {})
                message_date = message_data.get('date', '')[:10]  # Extract date part

                # Initialize daily counter if needed
                if message_date not in daily_action_counts:
                    daily_action_counts[message_date] = {
                        'name_mentions': 0, 'spells': 0, 'dialogue': 0,
                        'roll': 0, 'weapon': 0, 'no_action': 0
                    }

                for para_id, para_data in paragraphs.items():
                    if isinstance(para_data, dict):
                        action_counts['total_paragraphs'] += 1

                        # Count character labels
                        para_label = para_data.get('label', 'unlabeled')
                        if para_label == 'in-character':
                            action_counts['in_character_paragraphs'] += 1
                        elif para_label == 'out-of-character':
                            action_counts['out_of_character_paragraphs'] += 1
                        elif para_label == 'mixed':
                            action_counts['mixed_paragraphs'] += 1
                        else:
                            action_counts['unlabeled_paragraphs'] += 1

                        # Process actions if they exist
                        para_actions = para_data.get('actions', [])

                        if isinstance(para_actions, list):
                            if len(para_actions) == 0:
                                action_counts['no_action_paragraphs'] += 1
                                daily_action_counts[message_date]['no_action'] += 1
                            else:
                                # Track which action types this paragraph has (avoid double counting)
                                paragraph_action_types = set()

                                # Count each action type (paragraphs can have multiple types)
                                for action_type in para_actions:
                                    action_type_lower = str(action_type).lower()

                                    if 'name' in action_type_lower or 'mention' in action_type_lower:
                                        paragraph_action_types.add('name_mentions')
                                    elif 'spell' in action_type_lower or 'magic' in action_type_lower:
                                        paragraph_action_types.add('spells')
                                    elif 'dialogue' in action_type_lower or 'talk' in action_type_lower or 'speak' in action_type_lower:
                                        paragraph_action_types.add('dialogue')
                                    elif 'roll' in action_type_lower or 'dice' in action_type_lower or 'check' in action_type_lower:
                                        paragraph_action_types.add('roll')
                                    elif 'weapon' in action_type_lower or 'attack' in action_type_lower or 'fight' in action_type_lower:
                                        paragraph_action_types.add('weapon')

                                # If no recognized actions found, mark as no_action
                                if not paragraph_action_types:
                                    paragraph_action_types.add('no_action')

                                # Count each action type found in this paragraph
                                for action_type in paragraph_action_types:
                                    if action_type == 'name_mentions':
                                        action_counts['name_mentions_paragraphs'] += 1
                                        daily_action_counts[message_date]['name_mentions'] += 1
                                    elif action_type == 'spells':
                                        action_counts['spells_paragraphs'] += 1
                                        daily_action_counts[message_date]['spells'] += 1
                                    elif action_type == 'dialogue':
                                        action_counts['dialogue_paragraphs'] += 1
                                        daily_action_counts[message_date]['dialogue'] += 1
                                    elif action_type == 'roll':
                                        action_counts['roll_paragraphs'] += 1
                                        daily_action_counts[message_date]['roll'] += 1
                                    elif action_type == 'weapon':
                                        action_counts['weapon_paragraphs'] += 1
                                        daily_action_counts[message_date]['weapon'] += 1
                                    elif action_type == 'no_action':
                                        action_counts['no_action_paragraphs'] += 1
                                        daily_action_counts[message_date]['no_action'] += 1
                        else:
                            # Handle non-list actions (fallback)
                            action_counts['no_action_paragraphs'] += 1
                            daily_action_counts[message_date]['no_action'] += 1

            # Handle old format messages without paragraph structure
            else:
                action_counts['total_paragraphs'] += 1
                action_counts['unlabeled_paragraphs'] += 1
                action_counts['no_action_paragraphs'] += 1

                message_date = message_data.get('date', '')[:10]
                if message_date not in daily_action_counts:
                    daily_action_counts[message_date] = {
                        'name_mentions': 0, 'spells': 0, 'dialogue': 0,
                        'roll': 0, 'weapon': 0, 'no_action': 0
                    }
                daily_action_counts[message_date]['no_action'] += 1

    # Calculate percentages
    total_paragraphs = action_counts['total_paragraphs']
    if total_paragraphs > 0:
        action_counts.update({
            'name_mentions_percentage': (action_counts['name_mentions_paragraphs'] / total_paragraphs) * 100,
            'spells_percentage': (action_counts['spells_paragraphs'] / total_paragraphs) * 100,
            'dialogue_percentage': (action_counts['dialogue_paragraphs'] / total_paragraphs) * 100,
            'roll_percentage': (action_counts['roll_paragraphs'] / total_paragraphs) * 100,
            'weapon_percentage': (action_counts['weapon_paragraphs'] / total_paragraphs) * 100,
            'no_action_percentage': (action_counts['no_action_paragraphs'] / total_paragraphs) * 100,
            'in_character_percentage': (action_counts['in_character_paragraphs'] / total_paragraphs) * 100,
            'out_of_character_percentage': (action_counts['out_of_character_paragraphs'] / total_paragraphs) * 100
        })
    else:
        # Handle empty dataset
        action_counts.update({
            'name_mentions_percentage': 0, 'spells_percentage': 0, 'dialogue_percentage': 0,
            'roll_percentage': 0, 'weapon_percentage': 0, 'no_action_percentage': 0,
            'in_character_percentage': 0, 'out_of_character_percentage': 0
        })


    # Convert daily data to sorted list for time series analysis
    sorted_dates = sorted(daily_action_counts.keys())
    action_counts['daily_data'] = daily_action_counts
    action_counts['dates'] = sorted_dates

    return action_counts

