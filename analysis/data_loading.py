"""
D&D Data Loading Module

This module provides functions for loading and processing Dungeons & Dragons gameplay logs
from JSON format into pandas DataFrames. It handles both individual campaign files and
legacy single-file formats with memory-optimized loading strategies.

Functions:
    load_dnd_data: Convert nested D&D JSON data into a clean DataFrame
    load_all_campaigns: Load and process multiple campaigns from files
    _load_campaigns_from_individual_files: Memory-efficient loading from individual files
    _determine_primary_label: Helper function to determine primary message label
    _detect_dice_rolls: Helper function to detect dice rolls in messages
    _classify_message_type: Helper function to classify message types

Author: Claude Code Assistant
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import json

# ===================================================================
# SINGLE-CAMPAIGN Loading and processing
# ===================================================================


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

# ===================================================================
# MULTI-CAMPAIGN Loading
# ===================================================================


def load_all_campaigns(
    json_file_path: str,
    max_campaigns: Optional[int] = None,
    show_progress: bool = True,
    return_json: bool = False) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict]]:
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

    # Auto-detect whether to use individual files or legacy format
    json_path = Path(json_file_path)

    return _load_campaigns_from_individual_files(json_path,
                                                 max_campaigns, show_progress,
                                                 return_json)


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

    # Get sorted list of campaign files
    campaign_files = sorted([f for f in campaigns_dir.glob('*.json') if f.is_file()])

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

    if show_progress:
        print(f"✅ Successfully loaded {successful_campaigns} campaigns")
        print(f"📈 Total messages across all campaigns: {total_messages:,}")

    if return_json:
        return campaign_dataframes, json_data
    else:
        return campaign_dataframes
