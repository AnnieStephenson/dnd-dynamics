"""
Jaccard Similarity Analysis for D&D Campaigns

This module measures lexical cohesion in D&D campaigns using Jaccard similarity
to analyze lexical overlap patterns between players across game sessions.

Key Metric:
- Group Cohesion Score: Measures how well players are linguistically coordinating
  with each other within game sessions. Higher scores indicate better alignment.

For 2 players: Uses standard dyadic alignment between the pair
For 3+ players: Uses player-to-group alignment (each player vs rest of group)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from . import _cache
from .result import MetricResult


def analyze_jaccard(data: Dict[str, pd.DataFrame],
                    show_progress: bool = True,
                    cache_dir: Optional[str] = None,
                    force_refresh: bool = False) -> Dict[str, MetricResult]:
    """
    Calculate Jaccard similarity cohesion metrics for campaigns.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        show_progress: Whether to show progress for multi-campaign analysis
        cache_dir: Directory for caching results (defaults to data/processed/cohesion_results)
        force_refresh: Whether to force recomputation even if cached results exist

    Returns:
        Dict[campaign_id, MetricResult]
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")

    # Set default cache directory
    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'cohesion_results')

    # Handle caching
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "jaccard"
    )

    # Process missing campaigns
    new_results = {}
    if data_to_process:
        if show_progress and len(data_to_process) > 1:
            iterator = tqdm(data_to_process.items(), desc="Analyzing campaign cohesion", total=len(data_to_process))
        else:
            iterator = data_to_process.items()

        for campaign_id, df in iterator:
            new_results[campaign_id] = _analyze_single_campaign(
                df, campaign_id, show_progress=False
            )

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "jaccard"
    )


def _analyze_single_campaign(df: pd.DataFrame,
                             campaign_id: str,
                             show_progress: bool = False) -> Optional[MetricResult]:
    """
    Run Jaccard cohesion analysis for a single campaign using DataFrame.

    Args:
        df: Campaign DataFrame with 'text', 'player', and optionally 'session_id' columns
        campaign_id: Campaign identifier
        show_progress: Whether to show progress indicators

    Returns:
        MetricResult with cohesion analysis results or None if analysis failed
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
            cohesion_score, _ = _calculate_dyadic_cohesion(
                session_df, 'text', 'player')
        else:
            cohesion_score, _ = _calculate_group_cohesion(
                session_df, 'text', 'player')

        session_cohesion_scores.append(cohesion_score)

    if not session_cohesion_scores:
        print(f"No valid sessions found for cohesion analysis in {campaign_id}")
        return None

    # Per-player self-consistency across sessions and vocabulary size
    by_player = {}
    for player in df['player'].dropna().unique():
        mean_jaccard, pairwise_scores = _calculate_player_self_jaccard(df, player)

        # Calculate total vocabulary size for this player
        player_texts = df[df['player'] == player]['text'].dropna()
        player_vocab = set(' '.join(player_texts).lower().split())
        vocab_size = len(player_vocab)

        by_player[player] = MetricResult(
            series={'jaccard_session_pairs': pairwise_scores},
            summary={'mean_jaccard': mean_jaccard, 'vocabulary_size': vocab_size},
            by_player={},
            metadata={'session_count': int(df[df['player'] == player]['session_id'].nunique())}
        )

    return MetricResult(
        series={
            'jaccard_session': np.array(session_cohesion_scores),
        },
        summary={
            'mean_jaccard': float(np.mean(session_cohesion_scores)),
        },
        by_player=by_player,
        metadata={
            'campaign_id': campaign_id,
            'total_messages': len(df),
            'total_players': unique_players,
            'session_count': len(session_cohesion_scores),
        }
    )


def _calculate_dyadic_cohesion(session_df: pd.DataFrame,
                              text_col: str,
                              player_col: str) -> tuple[float, str]:
    """
    Calculate Jaccard similarity between two players.

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

    # Calculate Jaccard similarity between players
    player1_words = set(player1_combined.lower().split())
    player2_words = set(player2_combined.lower().split())

    if len(player1_words) == 0 or len(player2_words) == 0:
        alignment_score = 0.0
    else:
        intersection = len(player1_words & player2_words)
        union = len(player1_words | player2_words)
        alignment_score = intersection / union if union > 0 else 0.0

    return alignment_score, "dyadic"


def _calculate_player_self_jaccard(df: pd.DataFrame, player: str) -> tuple:
    """
    Calculate a player's vocabulary consistency across sessions.

    For each pair of sessions the player participated in, calculate
    Jaccard similarity between their vocabularies in those sessions.

    Args:
        df: Campaign DataFrame with 'text', 'player', 'session_id' columns
        player: Player name to analyze

    Returns:
        (mean_jaccard, pairwise_scores_array)
        Returns (np.nan, empty array) if player has fewer than 2 sessions
    """
    player_df = df[df['player'] == player]
    sessions = player_df['session_id'].dropna().unique()

    if len(sessions) < 2:
        return np.nan, np.array([])

    # Get vocabulary per session
    session_vocabs = {}
    for sess in sessions:
        sess_texts = player_df[player_df['session_id'] == sess]['text'].dropna()
        sess_text = ' '.join(sess_texts)
        session_vocabs[sess] = set(sess_text.lower().split())

    # Calculate pairwise Jaccard between sessions
    scores = []
    session_list = list(sessions)
    for i in range(len(session_list)):
        for j in range(i + 1, len(session_list)):
            vocab_a = session_vocabs[session_list[i]]
            vocab_b = session_vocabs[session_list[j]]
            if vocab_a and vocab_b:
                intersection = len(vocab_a & vocab_b)
                union = len(vocab_a | vocab_b)
                jaccard = intersection / union if union > 0 else 0.0
                scores.append(jaccard)

    if not scores:
        return np.nan, np.array([])

    return float(np.mean(scores)), np.array(scores)


def _calculate_group_cohesion(session_df: pd.DataFrame,
                             text_col: str,
                             player_col: str) -> tuple[float, str]:
    """
    Calculate player-to-group Jaccard similarity for 3+ players.

    Uses player-to-group approach: each player's similarity with the rest of the group,
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

        # Calculate Jaccard similarity between target player and others
        target_words = set(target_combined.lower().split())
        others_words = set(others_combined.lower().split())

        if len(target_words) == 0 or len(others_words) == 0:
            alignment_score = 0.0
        else:
            intersection = len(target_words & others_words)
            union = len(target_words | others_words)
            alignment_score = intersection / union if union > 0 else 0.0

        player_alignments.append(alignment_score)

    # Average alignment across all players for group cohesion score
    group_cohesion = np.mean(player_alignments)

    return group_cohesion, "player_to_group"
