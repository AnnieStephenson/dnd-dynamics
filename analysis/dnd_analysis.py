"""
D&D Gameplay Log Analysis Module

This module provides functions for analyzing Dungeons & Dragons gameplay logs
stored in JSON format. Each function accepts a pre-loaded DataFrame and returns
analysis results as data structures suitable for visualization.

Author: Claude Code Assistant
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

def load_dnd_data(json_data: Dict) -> pd.DataFrame:
    """
    Convert nested D&D JSON data into a clean DataFrame with label-aware processing.
    
    Args:
        json_data: Dictionary containing campaign data
        
    Returns:
        pd.DataFrame: Clean DataFrame with one row per message, including label-separated text
    """
    rows = []
    
    for campaign_id, messages in json_data.items():
        for message_id, message_data in messages.items():
            # Handle different text formats with label separation
            text_content = ''
            in_character_text = ''
            out_of_character_text = ''
            mixed_text = ''
            label_counts = {'in-character': 0, 'out-of-character': 0, 'mixed': 0, 'unlabeled': 0}
            
            if 'text' in message_data:
                # Old format: single text field (no label information)
                text_content = message_data.get('text', '')
                label_counts['unlabeled'] = 1
            elif 'paragraphs' in message_data:
                # New format: paragraphs with text segments and labels
                paragraphs = message_data.get('paragraphs', {})
                all_text_segments = []
                in_char_segments = []
                out_char_segments = []
                mixed_segments = []
                
                for para_id in sorted(paragraphs.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    para_data = paragraphs[para_id]
                    if isinstance(para_data, dict) and 'text' in para_data:
                        para_text = para_data['text']
                        all_text_segments.append(para_text)
                        
                        # Process by label
                        para_label = para_data.get('label', 'unlabeled')
                        if para_label == 'in-character':
                            in_char_segments.append(para_text)
                            label_counts['in-character'] += 1
                        elif para_label == 'out-of-character':
                            out_char_segments.append(para_text)
                            label_counts['out-of-character'] += 1
                        elif para_label == 'mixed':
                            mixed_segments.append(para_text)
                            label_counts['mixed'] += 1
                        else:
                            # Handle missing or unknown labels
                            label_counts['unlabeled'] += 1
                
                # Combine text by category
                text_content = ' '.join(all_text_segments)
                in_character_text = ' '.join(in_char_segments)
                out_of_character_text = ' '.join(out_char_segments)
                mixed_text = ' '.join(mixed_segments)
            
            # Extract actions from paragraphs if available
            actions = message_data.get('actions', {})
            if 'paragraphs' in message_data:
                paragraphs = message_data.get('paragraphs', {})
                for para_data in paragraphs.values():
                    if isinstance(para_data, dict) and 'actions' in para_data:
                        para_actions = para_data.get('actions', [])
                        if para_actions:  # If there are actions in this paragraph
                            # Merge paragraph actions into main actions dict
                            if not isinstance(actions, dict):
                                actions = {}
                            if 'paragraph_actions' not in actions:
                                actions['paragraph_actions'] = []
                            actions['paragraph_actions'].extend(para_actions)
                    
                    # Fallback classification for paragraphs without existing action labels
                    elif isinstance(para_data, dict) and 'text' in para_data and 'actions' not in para_data:
                        para_text = para_data['text'].lower()
                        action_type = None
                        
                        # Simple rule-based classification for unlabeled paragraphs
                        if 'roll' in para_text or 'd20' in para_text or 'dice' in para_text:
                            action_type = 'roll'
                        elif any(word in para_text for word in ['says', 'asks', 'replies', '"', "'"]):
                            action_type = 'dialogue'
                        elif any(word in para_text for word in ['spell', 'cast', 'magic']):
                            action_type = 'spells'
                        elif any(word in para_text for word in ['attack', 'sword', 'weapon', 'hit']):
                            action_type = 'weapon'
                        # Note: name_mentions would require more complex analysis
                        
                        # Add the classified action to the actions structure
                        if action_type:
                            if not isinstance(actions, dict):
                                actions = {}
                            if 'paragraph_actions' not in actions:
                                actions['paragraph_actions'] = []
                            actions['paragraph_actions'].append(action_type)
            
            row = {
                'campaign_id': campaign_id,
                'message_id': int(message_id),
                'date': pd.to_datetime(message_data.get('date')),
                'player': message_data.get('player'),
                'character': message_data.get('character'),
                'text': text_content,
                'in_character_text': in_character_text,
                'out_of_character_text': out_of_character_text,
                'mixed_text': mixed_text,
                'label_counts': label_counts,
                'in_combat': message_data.get('in_combat', False),
                'name_mentions': message_data.get('name_mentions', []),
                'actions': actions,
                'gender': message_data.get('gender'),
                'race': message_data.get('race'),
                'class': message_data.get('class'),
                'inventory': message_data.get('inventory', [])
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['campaign_id', 'date']).reset_index(drop=True)
    
    # Add derived columns
    df['word_count'] = df['text'].str.split().str.len().fillna(0)
    df['in_character_word_count'] = df['in_character_text'].str.split().str.len().fillna(0)
    df['out_of_character_word_count'] = df['out_of_character_text'].str.split().str.len().fillna(0)
    df['mixed_word_count'] = df['mixed_text'].str.split().str.len().fillna(0)
    df['has_dice_roll'] = df.apply(_detect_dice_rolls, axis=1)
    df['message_type'] = df.apply(_classify_message_type, axis=1)
    df['primary_label'] = df.apply(_determine_primary_label, axis=1)
    
    return df

def _determine_primary_label(row: pd.Series) -> str:
    """Helper function to determine the primary label for a message."""
    label_counts = row['label_counts']
    
    # Find the label with the most paragraphs
    max_count = 0
    primary_label = 'unlabeled'
    
    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            primary_label = label
    
    return primary_label

def _detect_dice_rolls(row: pd.Series) -> bool:
    """Helper function to detect dice rolls in message."""
    # Check actions field for rolls
    if 'check' in row['actions'] and 'roll' in row['actions']['check']:
        return True
    if 'survival' in row['actions'] and 'roll' in row['actions']['survival']:
        return True
    
    # Check text for dice notation patterns
    text = str(row['text']).lower()
    dice_patterns = [r'\d+d\d+', r'roll', r'\d{1,2}\s*(?:that|with)', r'rolled']
    return any(re.search(pattern, text) for pattern in dice_patterns)

def _classify_message_type(row: pd.Series) -> str:
    """Helper function to classify message as action or dialogue."""
    text = str(row['text']).lower()
    
    # Action keywords
    action_keywords = ['roll', 'check', 'move', 'attack', 'cast', 'look', 'search', 
                      'investigate', 'perception', 'stealth', 'climb', 'jump']
    
    # Dialogue indicators (quoted speech)
    has_quotes = '"' in row['text'] or "'" in row['text']
    
    # Check for action keywords
    has_action_words = any(keyword in text for keyword in action_keywords)
    
    if has_action_words or row['has_dice_roll']:
        return 'action'
    elif has_quotes:
        return 'dialogue'
    else:
        return 'narrative'

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


def analyze_action_vs_dialogue(df: pd.DataFrame) -> Dict:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use analyze_paragraph_actions() for detailed action type analysis.
    
    Analyze time series of action-related vs. dialogue-related posts.
    
    Args:
        df: DataFrame containing D&D messages
        
    Returns:
        Dict containing action/dialogue statistics and time series data
    """
    df_sorted = df.sort_values('date').copy()
    
    # Group by date and count message types
    daily_counts = df_sorted.groupby([df_sorted['date'].dt.date, 'message_type']).size().unstack(fill_value=0)
    
    # Calculate rolling averages to smooth the data
    window = min(7, len(daily_counts) // 3)  # 7-day window or 1/3 of data
    if window > 1:
        rolling_counts = daily_counts.rolling(window=window, center=True).mean()
    else:
        rolling_counts = daily_counts
    
    # Calculate overall statistics
    message_type_counts = df['message_type'].value_counts()
    total_posts = len(df)
    
    return {
        'total_posts': total_posts,
        'action_posts': message_type_counts.get('action', 0),
        'dialogue_posts': message_type_counts.get('dialogue', 0),
        'narrative_posts': message_type_counts.get('narrative', 0),
        'action_percentage': (message_type_counts.get('action', 0) / total_posts) * 100,
        'dialogue_percentage': (message_type_counts.get('dialogue', 0) / total_posts) * 100,
        'daily_data': daily_counts,
        'rolling_data': rolling_counts,
        'dates': rolling_counts.index if len(rolling_counts) > 0 else [],
        'message_types': rolling_counts.columns.tolist() if len(rolling_counts) > 0 else []
    }

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
            'total_posts': total_posts,
            'campaign_duration_days': date_range.days,
            'unique_players': unique_players,
            'unique_characters': unique_characters,
            'posts_per_day': total_posts / max(date_range.days, 1),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        },
        'posting_patterns': {
            'most_active_player': df['player'].value_counts().index[0] if len(df) > 0 else None,
            'posts_by_most_active': df['player'].value_counts().iloc[0] if len(df) > 0 else 0,
            'average_post_length': df['word_count'].mean(),
            'longest_post_words': df['word_count'].max()
        },
        'gameplay_characteristics': {
            'dice_roll_percentage': (df['has_dice_roll'].sum() / total_posts) * 100,
            'action_percentage': (df['message_type'].value_counts().get('action', 0) / total_posts) * 100,
            'dialogue_percentage': (df['message_type'].value_counts().get('dialogue', 0) / total_posts) * 100,
            'combat_posts': df['in_combat'].sum(),
            'combat_percentage': (df['in_combat'].sum() / total_posts) * 100
        }
    }
    
    return report


# ===================================================================
# MULTI-CAMPAIGN ANALYSIS FUNCTIONS
# ===================================================================

def load_all_campaigns(json_file_path: str, max_campaigns: Optional[int] = None, 
                      show_progress: bool = True, return_json: bool = False) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict]]:
    """
    Load and process multiple campaigns from individual campaign files.
    
    MEMORY OPTIMIZED: Loads individual campaign files instead of entire JSON file.
    
    Args:
        json_file_path: Path to campaigns directory or legacy JSON file (auto-detects)
        max_campaigns: Maximum number of campaigns to load (None for all)
        show_progress: Whether to show progress indicators
        return_json: If True, return both DataFrames and original JSON data
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping campaign_id to DataFrame
        OR Tuple[Dict[str, pd.DataFrame], Dict]: (DataFrames, JSON data) if return_json=True
    """
    import json
    
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not available, progress bars disabled")
            show_progress = False
            tqdm = None
    else:
        tqdm = None
    
    # Auto-detect whether to use individual files or legacy format
    json_path = Path(json_file_path)
    
    # Check if individual campaigns directory exists
    if json_path.name == 'data-labels.json':
        # Legacy path - look for individual campaigns in same directory structure
        individual_campaigns_dir = json_path.parent / 'individual_campaigns'
    elif json_path.is_dir():
        # Direct directory path
        individual_campaigns_dir = json_path
    else:
        # Assume it's a path to the data directory
        individual_campaigns_dir = json_path.parent / 'individual_campaigns'
    
    # Try to use individual campaign files first (more memory efficient)
    if individual_campaigns_dir.exists() and individual_campaigns_dir.is_dir():
        return _load_campaigns_from_individual_files(
            individual_campaigns_dir, max_campaigns, show_progress, return_json
        )
    else:
        # Fallback to legacy JSON file loading
        if show_progress:
            print(f"⚠️  Individual campaigns directory not found: {individual_campaigns_dir}")
            print(f"📁 Falling back to legacy JSON file loading from {json_file_path}")
        
        return _load_campaigns_from_json_file(
            json_file_path, max_campaigns, show_progress, return_json
        )


def _load_campaigns_from_individual_files(campaigns_dir: Path, max_campaigns: Optional[int], 
                                        show_progress: bool, return_json: bool) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict]]:
    """
    Memory-efficient loading from individual campaign files.
    
    Args:
        campaigns_dir: Path to directory containing individual campaign JSON files
        max_campaigns: Maximum number of campaigns to load (None for all)
        show_progress: Whether to show progress indicators
        return_json: If True, return both DataFrames and original JSON data
        
    Returns:
        Campaign DataFrames and optionally JSON data
    """
    import json
    
    # Import tqdm for progress bars
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not available, progress bars disabled")
            show_progress = False
            tqdm = None
    else:
        tqdm = None
    
    # Get sorted list of campaign files
    campaign_files = sorted([f for f in campaigns_dir.glob('*.json') if f.is_file()])
    
    if not campaign_files:
        if show_progress:
            print(f"❌ No campaign files found in {campaigns_dir}")
        return {} if not return_json else ({}, {})
    
    total_available = len(campaign_files)
    campaigns_to_load = min(max_campaigns, total_available) if max_campaigns is not None else total_available
    
    if show_progress:
        print(f"📂 Loading campaigns from individual files in {campaigns_dir}")
        print(f"📊 Found {total_available} campaign files")
        if max_campaigns is not None:
            print(f"🎯 Loading first {campaigns_to_load} campaigns (max_campaigns={max_campaigns})")
    
    # Process only the required number of files
    files_to_process = campaign_files[:campaigns_to_load]
    campaign_dataframes = {}
    json_data = {}
    
    # Progress indicator
    if show_progress and len(files_to_process) > 1:
        iterator = tqdm(files_to_process, desc="Loading campaigns")
    else:
        iterator = files_to_process
    
    successful_campaigns = 0
    total_messages = 0
    
    for campaign_file in iterator:
        try:
            # Extract campaign ID from filename (without extension)
            campaign_id = campaign_file.stem
            
            # Load individual campaign file
            with open(campaign_file, 'r', encoding='utf-8') as f:
                campaign_data = json.load(f)
            
            # Create single-campaign data structure for load_dnd_data
            single_campaign_data = {campaign_id: campaign_data}
            
            # Load using existing function
            df = load_dnd_data(single_campaign_data)
            
            # Only keep non-empty campaigns
            if len(df) > 0:
                campaign_dataframes[campaign_id] = df
                total_messages += len(df)
                successful_campaigns += 1
                
                # Store JSON data if requested
                if return_json:
                    json_data[campaign_id] = campaign_data
                    
        except Exception as e:
            if show_progress:
                print(f"⚠️  Warning: Failed to process campaign {campaign_file.name}: {e}")
            continue
    
    if show_progress:
        print(f"✅ Successfully loaded {successful_campaigns} campaigns")
        print(f"📈 Total messages across all campaigns: {total_messages:,}")
        
        # Memory efficiency report
        if max_campaigns is not None and max_campaigns < total_available:
            memory_saved = total_available - campaigns_to_load
            print(f"💾 Memory optimization: Avoided loading {memory_saved} unnecessary campaigns")
    
    if return_json:
        return campaign_dataframes, json_data
    else:
        return campaign_dataframes


def _load_campaigns_from_json_file(json_file_path: str, max_campaigns: Optional[int], 
                                 show_progress: bool, return_json: bool) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict]]:
    """
    Legacy loading from single large JSON file.
    
    Args:
        json_file_path: Path to JSON file containing multiple campaigns
        max_campaigns: Maximum number of campaigns to load (None for all)
        show_progress: Whether to show progress indicators
        return_json: If True, return both DataFrames and original JSON data
        
    Returns:
        Campaign DataFrames and optionally JSON data
    """
    import json
    
    # Import tqdm for progress bars
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not available, progress bars disabled")
            show_progress = False
            tqdm = None
    else:
        tqdm = None
    
    if show_progress:
        print(f"📁 Loading campaigns from JSON file: {json_file_path}")
    
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        all_campaigns_data = json.load(f)
    
    total_campaigns = len(all_campaigns_data)
    if show_progress:
        print(f"📊 Found {total_campaigns} campaigns")
    
    # Limit campaigns if specified
    campaign_ids = list(all_campaigns_data.keys())
    if max_campaigns is not None:
        campaign_ids = campaign_ids[:max_campaigns]
        if show_progress:
            print(f"🎯 Processing first {len(campaign_ids)} campaigns")
    
    # Process each campaign
    campaign_dataframes = {}
    if show_progress and len(campaign_ids) > 1:
        iterator = tqdm(campaign_ids, desc="Processing campaigns")
    else:
        iterator = campaign_ids
    
    for campaign_id in iterator:
        try:
            # Create single-campaign data structure
            single_campaign_data = {campaign_id: all_campaigns_data[campaign_id]}
            
            # Load using existing function
            df = load_dnd_data(single_campaign_data)
            
            # Only keep non-empty campaigns
            if len(df) > 0:
                campaign_dataframes[campaign_id] = df
                
        except Exception as e:
            if show_progress:
                print(f"⚠️  Warning: Failed to process campaign {campaign_id}: {e}")
            continue
    
    if show_progress:
        successful_campaigns = len(campaign_dataframes)
        print(f"✅ Successfully loaded {successful_campaigns} campaigns")
        
        # Show basic stats
        total_messages = sum(len(df) for df in campaign_dataframes.values())
        print(f"📈 Total messages across all campaigns: {total_messages:,}")
    
    if return_json:
        # Return filtered JSON data for successful campaigns only
        filtered_json_data = {cid: all_campaigns_data[cid] for cid in campaign_dataframes.keys()}
        return campaign_dataframes, filtered_json_data
    else:
        return campaign_dataframes


def analyze_all_campaigns(campaign_dataframes: Dict[str, pd.DataFrame], 
                         original_json_data: Optional[Dict] = None,
                         show_progress: bool = True) -> Dict[str, Dict]:
    """
    Apply all analysis functions across multiple campaigns.
    
    Args:
        campaign_dataframes: Dictionary of campaign DataFrames from load_all_campaigns()
        original_json_data: Optional original JSON data for paragraph-level action analysis
        show_progress: Whether to show progress indicators
        
    Returns:
        Dict containing per-campaign and aggregated results for all metrics
    """
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not available, progress bars disabled")
            show_progress = False
            tqdm = None
    else:
        tqdm = None
    
    # Initialize results structure
    results = {
        'per_campaign': {},
        'aggregated': {},
        'summary_stats': {}
    }
    
    if show_progress:
        print(f"Analyzing {len(campaign_dataframes)} campaigns...")
    
    # Process each campaign
    iterator = tqdm(campaign_dataframes.items(), desc="Analyzing campaigns") if show_progress else campaign_dataframes.items()
    
    for campaign_id, df in iterator:
        try:
            # Run all analysis functions for this campaign
            campaign_results = {
                'time_intervals_overall': analyze_time_intervals(df, by_player=False),
                'time_intervals_by_player': analyze_time_intervals(df, by_player=True),
                'cumulative_posts_overall': analyze_cumulative_posts(df, by_player=False),
                'cumulative_posts_by_player': analyze_cumulative_posts(df, by_player=True),
                'unique_players_characters': analyze_unique_players_characters(df),
                'post_lengths_overall': analyze_post_lengths(df, by_player=False),
                'post_lengths_by_player': analyze_post_lengths(df, by_player=True),
                'post_lengths_by_label_overall': analyze_post_lengths_by_label(df, by_player=False),
                'post_lengths_by_label_by_player': analyze_post_lengths_by_label(df, by_player=True),
                'action_vs_dialogue': analyze_action_vs_dialogue(df),
                'character_mentions': analyze_character_mentions(df),
                'dice_roll_frequency': analyze_dice_roll_frequency(df),
                'summary_report': generate_summary_report(df)
            }
            
            # Add paragraph-level action analysis if original JSON data is available
            if original_json_data and campaign_id in original_json_data:
                single_campaign_data = {campaign_id: original_json_data[campaign_id]}
                campaign_results['paragraph_actions'] = analyze_paragraph_actions(single_campaign_data)
            
            results['per_campaign'][campaign_id] = campaign_results
            
        except Exception as e:
            if show_progress:
                print(f"Warning: Failed to analyze campaign {campaign_id}: {e}")
            continue
    
    # Generate aggregated results
    if show_progress:
        print("Generating aggregated statistics...")
    
    results['aggregated'] = aggregate_campaign_metrics(results['per_campaign'])
    results['summary_stats'] = generate_multi_campaign_summary(campaign_dataframes, results['per_campaign'])
    
    return results


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
    
    # Aggregate action vs dialogue percentages
    total_action = sum(per_campaign_results[cid]['action_vs_dialogue']['action_posts'] 
                      for cid in campaign_ids)
    total_dialogue = sum(per_campaign_results[cid]['action_vs_dialogue']['dialogue_posts'] 
                        for cid in campaign_ids)
    total_narrative = sum(per_campaign_results[cid]['action_vs_dialogue']['narrative_posts'] 
                         for cid in campaign_ids)
    total_posts = total_action + total_dialogue + total_narrative
    
    if total_posts > 0:
        aggregated['message_types'] = {
            'action_posts': total_action,
            'dialogue_posts': total_dialogue,
            'narrative_posts': total_narrative,
            'total_posts': total_posts,
            'action_percentage': (total_action / total_posts) * 100,
            'dialogue_percentage': (total_dialogue / total_posts) * 100,
            'narrative_percentage': (total_narrative / total_posts) * 100
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


# ===================================================================
# INTELLIGENT CACHING SYSTEM
# ===================================================================

import pickle
import os
import hashlib
from datetime import datetime
from pathlib import Path


def _get_data_file_hash(data_file_path: str) -> str:
    """Calculate hash of data file for cache validation."""
    if not os.path.exists(data_file_path):
        return ""
    
    # Use file size and modification time for quick validation
    stat = os.stat(data_file_path)
    hash_string = f"{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_string.encode()).hexdigest()


def save_campaign_results(all_results: Dict, num_campaigns: int, 
                         data_file_path: str = 'Game-Data/data-labels.json',
                         cache_dir: str = 'campaign_stats_cache') -> str:
    """
    Save campaign analysis results to cache with metadata.
    
    Args:
        all_results: Complete results from analyze_all_campaigns()
        num_campaigns: Number of campaigns analyzed
        data_file_path: Path to source data file for validation
        cache_dir: Directory to store cache files
        
    Returns:
        str: Path to saved cache file
    """
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
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
        campaign_dfs, json_data = load_all_campaigns(data_file_path, max_campaigns, show_progress, return_json=True)
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
        
        campaign_dfs, json_data = load_all_campaigns(data_file_path, max_campaigns, show_progress, return_json=True)
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
    all_campaign_dfs, all_json_data = load_all_campaigns(data_file_path, max_campaigns, show_progress=False, return_json=True)
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