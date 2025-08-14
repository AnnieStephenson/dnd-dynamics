"""
Group Cohesion Analysis for D&D Campaigns using Lexical Alignment

This module measures group linguistic cohesion in D&D campaigns using the ALIGN package
to analyze lexical alignment patterns between players across game sessions.

Key Metric:
- Group Cohesion Score: Measures how well players are linguistically coordinating
  with each other within game sessions. Higher scores indicate better alignment.

For 2 players: Uses standard dyadic alignment between the pair
For 3+ players: Uses player-to-group alignment (each player vs rest of group)

Author: Claude Code Assistant
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import align

# Download required NLTK data for ALIGN
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from . import data_loading as dl
from . import batch


def analyze_cohesion(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    show_progress: bool = True,
                    cache_dir: Optional[str] = None,
                    force_refresh: bool = False,
                    messages_per_session: int = 20) -> Union[Dict, Dict[str, Dict]]:
    """
    Calculate group cohesion metrics for single or multiple campaigns using DataFrames.
    
    Args:
        data: Single DataFrame or dict of DataFrames {campaign_id: df}
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/cohesion_results)
        force_refresh: Whether to force recomputation even if cached results exist
        messages_per_session: Number of messages per session for LLM campaigns (default: 20)
        
    Returns:
        Dict of cohesion results for single campaign, or Dict[campaign_id, results] for multiple
    """
    if isinstance(data, pd.DataFrame):
        # Single campaign analysis - no caching for single campaigns
        campaign_id = "not specified"
        return _analyze_single_campaign_cohesion(data, campaign_id, show_progress=False, messages_per_session=messages_per_session)
    
    elif isinstance(data, dict):
        # Multiple campaign analysis with caching support
        
        # Set default cache directory
        if cache_dir is None:
            repo_root = Path(__file__).parent.parent
            cache_dir = str(repo_root / 'data' / 'processed' / 'cohesion_results')
        
        # Handle caching using helper function
        cached_results, data_to_process = batch.handle_multi_campaign_caching(
            data, cache_dir, force_refresh, show_progress, "cohesion"
        )
        
        # Process missing campaigns
        new_results = {}
        if data_to_process:
            if show_progress and len(data_to_process) > 1:
                iterator = tqdm(data_to_process.items(), desc="Analyzing campaign cohesion", total=len(data_to_process))
            else:
                iterator = data_to_process.items()
            
            for campaign_id, df in iterator:
                new_results[campaign_id] = _analyze_single_campaign_cohesion(
                    df, campaign_id, show_progress=False, messages_per_session=messages_per_session
                )
        
        # Save new results and combine with cached results
        return batch.save_new_results_and_combine(
            cached_results, new_results, cache_dir, show_progress, "cohesion"
        )
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.DataFrame or Dict[str, pd.DataFrame]")


def _analyze_single_campaign_cohesion(df: pd.DataFrame, 
                                     campaign_id: str, 
                                     show_progress: bool = False,
                                     messages_per_session: int = 20) -> Optional[Dict]:
    """
    Run cohesion analysis for a single campaign using DataFrame.
    
    Args:
        df: Campaign DataFrame with 'text', 'player', and optionally 'session_id' columns
        campaign_id: Campaign identifier
        show_progress: Whether to show progress indicators
        messages_per_session: Number of messages per session for LLM campaigns
        
    Returns:
        Dict with cohesion analysis results or None if analysis failed
    """
    # Check required columns
    required_cols = ['text', 'player']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns for {campaign_id}: {missing_cols}")
        return None
    
    # Check minimum player requirement
    unique_players = df['player'].nunique()
    if unique_players < 2:
        print(f"Campaign {campaign_id} has only {unique_players} player(s). Need at least 2 for cohesion analysis.")
        return None
    
    # Use existing session_id column (created at data loading time)
    session_col = 'session_id'
    
    # Remove rows with missing data
    df_clean = df[['text', 'player', session_col]].dropna()
    if df_clean.empty:
        print(f"No valid data for {campaign_id} after removing missing values.")
        return None
    
    session_cohesion_scores = []
    session_metadata = []
    
    sessions = df_clean[session_col].unique()
    sessions = [s for s in sessions if not pd.isna(s)]
    
    for session_id in sessions:
        session_df = df_clean[df_clean[session_col] == session_id]
        
        # Skip sessions with insufficient data
        if len(session_df) < 2:
            continue
            
        session_players = session_df['player'].unique()
        if len(session_players) < 2:
            continue
        
        # Calculate cohesion based on player count
        if len(session_players) == 2:
            cohesion_score, alignment_type = _calculate_dyadic_cohesion(
                session_df, 'text', 'player')
        else:
            cohesion_score, alignment_type = _calculate_group_cohesion(
                session_df, 'text', 'player')
        
        session_cohesion_scores.append(cohesion_score)
        session_metadata.append({
            'session_id': session_id,
            'player_count': len(session_players),
            'message_count': len(session_df),
            'alignment_type': alignment_type
        })
    
    if not session_cohesion_scores:
        print(f"No valid sessions found for cohesion analysis in {campaign_id}")
        return None
    
    cohesion_analysis = {
        'session_cohesion_scores': session_cohesion_scores,
        'session_metadata': session_metadata,
        'session_count': len(session_cohesion_scores)
    }
    
    # Add campaign metadata
    cohesion_analysis['metadata'] = {
        'campaign_id': campaign_id,
        'total_messages': len(df),
        'total_players': unique_players
    }
    
    return cohesion_analysis


def _calculate_dyadic_cohesion(session_df: pd.DataFrame, 
                              text_col: str, 
                              player_col: str) -> tuple[float, str]:
    """
    Calculate lexical alignment between two players using ALIGN package.
    
    Parameters
    ----------
    session_df : pd.DataFrame
        Session data with exactly 2 players
    text_col : str
        Column name for message text
    player_col : str
        Column name for player identifiers
        
    Returns
    -------
    tuple[float, str]
        (cohesion_score, "dyadic")
    """
    players = session_df[player_col].unique()
    
    # Split messages by player
    player1_texts = session_df[session_df[player_col] == players[0]][text_col].tolist()
    player2_texts = session_df[session_df[player_col] == players[1]][text_col].tolist()
    
    # Combine texts into single strings per player for alignment calculation
    player1_combined = " ".join(player1_texts)
    player2_combined = " ".join(player2_texts)
    
    # PLACEHOLDER: Simple lexical overlap as proxy for alignment
    # TODO: Replace with proper ALIGN package integration once version compatibility is resolved
    
    # Calculate simple lexical overlap between players
    player1_words = set(player1_combined.lower().split())
    player2_words = set(player2_combined.lower().split())
    
    if len(player1_words) == 0 or len(player2_words) == 0:
        alignment_score = 0.0
    else:
        # Jaccard similarity as alignment proxy
        intersection = len(player1_words & player2_words)
        union = len(player1_words | player2_words)
        alignment_score = intersection / union if union > 0 else 0.0
    
    return alignment_score, "dyadic"


def _calculate_group_cohesion(session_df: pd.DataFrame,
                             text_col: str, 
                             player_col: str) -> tuple[float, str]:
    """
    Calculate player-to-group lexical alignment for 3+ players.
    
    Uses player-to-group approach: each player's alignment with the rest of the group,
    then averages across all players for overall group cohesion.
    
    Parameters
    ----------
    session_df : pd.DataFrame
        Session data with 3+ players
    text_col : str
        Column name for message text  
    player_col : str
        Column name for player identifiers
        
    Returns
    -------
    tuple[float, str]
        (cohesion_score, "player_to_group")
    """
    players = session_df[player_col].unique()
    
    player_alignments = []
    
    for target_player in players:
        # Get target player's messages
        target_texts = session_df[session_df[player_col] == target_player][text_col].tolist()
        
        # Get other players' messages  
        other_texts = session_df[session_df[player_col] != target_player][text_col].tolist()
        
        # Skip players with no messages
        if len(target_texts) == 0 or len(other_texts) == 0:
            continue
            
        # Combine texts
        target_combined = " ".join(target_texts)
        others_combined = " ".join(other_texts)
        
        # PLACEHOLDER: Simple lexical overlap as proxy for alignment 
        # TODO: Replace with proper ALIGN package integration once version compatibility is resolved
        
        # Calculate simple lexical overlap between target player and others
        target_words = set(target_combined.lower().split())
        others_words = set(others_combined.lower().split())
        
        if len(target_words) == 0 or len(others_words) == 0:
            alignment_score = 0.0
        else:
            # Jaccard similarity as alignment proxy
            intersection = len(target_words & others_words)
            union = len(target_words | others_words)
            alignment_score = intersection / union if union > 0 else 0.0
            
        player_alignments.append(alignment_score)
    
    # Average alignment across all players for group cohesion score
    group_cohesion = np.mean(player_alignments)
    
    return group_cohesion, "player_to_group"


