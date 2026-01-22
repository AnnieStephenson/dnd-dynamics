"""
D&D Data Loading Module

This module provides functions for loading and processing Dungeons & Dragons gameplay logs
from JSON format into pandas DataFrames. It handles both individual campaign files and
legacy single-file formats with memory-optimized loading strategies.

Functions:
    load_campaigns: Load campaigns from various sources (unified interface)
    _load_dnd_data: Convert nested D&D JSON data into a clean DataFrame (private helper)
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
from dnd_dynamics import config
from . import data_correction

# ===================================================================
# USER-FACING CAMPAIGN LOADING FUNCTIONS
# ===================================================================


def load_campaigns(source: Union[str, List[str], Path], max_campaigns: Optional[int] = None,
    show_progress: bool = True,
    return_json: bool = False,
    messages_per_session: int = None,
    filter_by: Optional[Dict] = None,
    min_turns: int = 5
) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict]]:
    """
    Load and process campaigns from various sources.

    Args:
        source: 'human' for human games, 'llm' for LLM games,
                Path/str for custom directory path OR single campaign name,
                List[str] for multiple campaign names
        max_campaigns: Maximum number of campaigns to load (None for all)
        show_progress: Whether to show progress indicators
        return_json: If True, return both DataFrames and original JSON data
        messages_per_session: Number of messages per session for creating session_id column.
                              If None, uses config.MESSAGES_PER_SESSION
        filter_by: Dict of metadata fields to filter on (LLM games only),
                   e.g. {'model': 'gpt-4o', 'summary_chunk_size': 50}
        min_turns: Minimum number of turns required to include a campaign (default 5)

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping campaign_id to DataFrame
        OR Tuple[Dict[str, pd.DataFrame], Dict]: (DataFrames, JSON data) if return_json=True
    """
    if messages_per_session is None:
        messages_per_session = config.MESSAGES_PER_SESSION

    module_dir = Path(__file__).parent.parent.parent  # Go up to repository root

    # Handle different source types
    if isinstance(source, list):
        # Multiple campaign names - search in standard directories
        return _load_campaigns_by_names(source, module_dir, return_json, messages_per_session, min_turns)

    elif isinstance(source, (str, Path)):
        source_str = str(source)

        # Check for special keywords
        if source_str == 'human':
            campaigns_dir = module_dir / 'data/corrected-human-games/'

            # Failsafe: auto-correct if folder empty/missing
            if not campaigns_dir.exists() or not any(campaigns_dir.glob('*.json')):
                if show_progress:
                    print("Corrected campaigns not found. Running corrections...")
                data_correction.correct_all_campaigns()
        elif source_str == 'llm':
            campaigns_dir = module_dir / 'data/llm-games/game-logs/'
        else:
            # Check if it's a directory path or a single campaign name
            potential_path = Path(source_str)
            if potential_path.is_absolute():
                campaigns_dir = potential_path
            else:
                # Check if it's a relative directory that exists
                relative_path = module_dir / source_str
                if relative_path.is_dir():
                    campaigns_dir = relative_path
                else:
                    # Treat as single campaign name
                    return _load_campaigns_by_names([source_str], module_dir, return_json, messages_per_session, min_turns)

        # Load from directory path
        return _load_campaigns_from_directory(campaigns_dir, max_campaigns, show_progress, return_json, messages_per_session, filter_by, min_turns)

    else:
        raise ValueError(f"Unsupported source type: {type(source)}")


def _load_campaigns_by_names(campaign_names: List[str], module_dir: Path, return_json: bool, messages_per_session: int, min_turns: int):
    """Load campaigns by searching for specific names in standard directories."""
    search_dirs = [
        module_dir / 'data/corrected-human-games/',
        module_dir / 'data/llm-games/game-logs/'
    ]

    campaign_dataframes = {}
    json_data = {}

    for campaign_name in campaign_names:
        # Search in each directory for matching JSON file
        for search_dir in search_dirs:
            campaign_file = search_dir / f"{campaign_name}.json"

            if campaign_file.exists():
                # Load the campaign file
                with open(campaign_file, 'r', encoding='utf-8') as f:
                    campaign_data = json.load(f)

                # Create single-campaign data structure for _load_dnd_data
                single_campaign_data = {campaign_name: campaign_data}

                # Load using helper function
                df = _load_dnd_data(single_campaign_data, messages_per_session)

                # Only keep campaigns with enough turns
                if len(df) >= min_turns:
                    campaign_dataframes[campaign_name] = df
                    if return_json:
                        json_data[campaign_name] = campaign_data
                    break

    if return_json:
        return campaign_dataframes, json_data
    else:
        return campaign_dataframes


def _load_campaigns_from_directory(campaigns_dir: Path, max_campaigns: Optional[int],
                                 show_progress: bool, return_json: bool, messages_per_session: int,
                                 filter_by: Optional[Dict] = None, min_turns: int = 5):
    """Load campaigns from a directory path."""
    # Check if this is the LLM games directory and filtering is requested
    llm_games_dir = campaigns_dir.parent  # game-logs/ -> llm-games/
    is_llm_dir = 'llm-games' in str(campaigns_dir)

    # If filtering LLM games, use metadata index
    if filter_by and is_llm_dir:
        index_path = llm_games_dir / 'metadata_index.json'
        if not index_path.exists():
            # Auto-rebuild index if missing
            if show_progress:
                print("Metadata index not found, rebuilding...")
            rebuild_metadata_index(llm_games_dir)

        if index_path.exists():
            with open(index_path, 'r') as f:
                metadata_index = json.load(f)

            # Filter to matching files
            matching_files = []
            for filename_base, metadata in metadata_index.items():
                matches = all(
                    (metadata.get(k) in v if isinstance(v, list) else metadata.get(k) == v)
                    for k, v in filter_by.items()
                )
                if matches:
                    file_path = campaigns_dir / f"{filename_base}.json"
                    if file_path.exists():
                        matching_files.append(file_path)

            campaign_files = sorted(matching_files)
            if show_progress:
                print(f"Filter matched {len(campaign_files)} campaigns")
        else:
            campaign_files = sorted([f for f in campaigns_dir.glob('*.json') if f.is_file()])
    else:
        # Get sorted list of campaign files
        campaign_files = sorted([f for f in campaigns_dir.glob('*.json') if f.is_file()])

    total_available = len(campaign_files)
    campaigns_to_load = min(max_campaigns, total_available) if max_campaigns is not None else total_available

    if show_progress:
        print(f"Loading {campaigns_to_load} campaigns from {campaigns_dir}")

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

        # Create single-campaign data structure for _load_dnd_data
        single_campaign_data = {campaign_id: campaign_data}

        # Load using existing function
        df = _load_dnd_data(single_campaign_data, messages_per_session)

        # Only keep campaigns with enough turns
        if len(df) >= min_turns:
            campaign_dataframes[campaign_id] = df
            total_messages += len(df)
            successful_campaigns += 1

            # Store JSON data if requested
            if return_json:
                json_data[campaign_id] = campaign_data

    if show_progress:
        print(f"Loaded {successful_campaigns} campaigns ({total_messages:,} messages)")

    if return_json:
        return campaign_dataframes, json_data
    else:
        return campaign_dataframes


# ===================================================================
# SINGLE-CAMPAIGN Loading and processing
# ===================================================================


def _load_dnd_data(json_data: Dict, messages_per_session: int = 5) -> pd.DataFrame:
    """
    Convert nested D&D JSON data into a clean DataFrame with label-aware processing.
    
    Args:
        json_data: Dictionary containing campaign data
        messages_per_session: Number of messages per session for creating session_id column
        
    Returns:
        pd.DataFrame: Clean DataFrame with one row per message, including label-separated text
    """
    rows = []

    for campaign_id, messages in json_data.items():
        for message_id, message_data in messages.items():
            # Skip metadata keys
            if message_id.startswith('_'):
                continue

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
    
    # Add session_id column based on fixed message blocks
    df['session_id'] = _create_block_sessions(df, messages_per_session)
    
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

def _create_block_sessions(df: pd.DataFrame, messages_per_session: int = 5) -> pd.Series:
    """
    Create session IDs based on fixed blocks of messages.
    
    Parameters
    ----------
    df : pd.DataFrame
        Campaign DataFrame
    messages_per_session : int
        Number of messages per session block
        
    Returns
    -------
    pd.Series
        Session IDs (e.g., 'session_1', 'session_2', etc.)
    """
    num_messages = len(df)
    session_ids = []
    
    for i in range(num_messages):
        session_num = i // messages_per_session + 1
        session_ids.append(f"session_{session_num}")
    
    return pd.Series(session_ids, index=df.index)


# ===================================================================
# METADATA INDEX FUNCTIONS
# ===================================================================


def rebuild_metadata_index(llm_games_dir: Path = None):
    """
    Rebuild metadata_index.json by scanning all campaign JSON files.

    Args:
        llm_games_dir: Path to llm-games directory (default: {project_root}/data/llm-games)

    Scans all game-logs/*.json files, reads _metadata from each, writes index.
    """
    if llm_games_dir is None:
        module_dir = Path(__file__).parent.parent.parent  # Go up to repository root
        llm_games_dir = module_dir / 'data' / 'llm-games'

    game_logs_dir = llm_games_dir / 'game-logs'
    if not game_logs_dir.exists():
        print(f"⚠️  No game-logs directory found at {game_logs_dir}")
        return

    metadata_index = {}
    campaign_files = sorted(game_logs_dir.glob('*.json'))

    for campaign_file in campaign_files:
        filename_base = campaign_file.stem
        with open(campaign_file, 'r', encoding='utf-8') as f:
            campaign_data = json.load(f)

        if '_metadata' in campaign_data:
            metadata_index[filename_base] = campaign_data['_metadata']

    index_path = llm_games_dir / 'metadata_index.json'
    with open(index_path, 'w') as f:
        json.dump(metadata_index, f, indent=2)

    print(f"📋 Rebuilt metadata index with {len(metadata_index)} entries: {index_path}")


# ===================================================================
# MULTI-CAMPAIGN Loading
# ===================================================================


