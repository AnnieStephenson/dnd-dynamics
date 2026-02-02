"""
Shared utilities for LLM social metrics.

These functions are used across multiple metrics (humor, conflict, cooperation,
norms, collab_creativity) to avoid code duplication.
"""

import re
import pandas as pd
from typing import Dict, List, Any


def format_turns(df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
    """
    Format DataFrame rows as readable text with turn numbers and character names.

    Output format:
    Turn 45 - CharacterName: message text here...
    Turn 46 - OtherCharacter: their message text...

    Args:
        df: Campaign DataFrame with 'text' and 'character' columns
        start_idx: Starting index (inclusive)
        end_idx: Ending index (inclusive)

    Returns:
        Formatted string of turns
    """
    lines = []
    for idx in range(start_idx, min(end_idx + 1, len(df))):
        row = df.iloc[idx]
        character = row.get('character', 'Unknown')
        text = row.get('text', '')
        lines.append(f"Turn {idx} - {character}: {text}")
    return '\n'.join(lines)


def parse_turn_extraction_response(
    response_text: str,
    window_start: int,
    window_end: int,
    no_match_token: str
) -> List[Dict]:
    """
    Parse LLM response for per-turn identification.

    Expects format:
    TURN 45: explanation text here
    TURN 47: another explanation
    ...

    Args:
        response_text: Raw LLM response
        window_start: Start of the extraction window (for validation)
        window_end: End of the extraction window (for validation)
        no_match_token: Token indicating no matches found (e.g., "NO_HUMOR_FOUND")

    Returns:
        List of dicts with 'turn_number' and 'explanation'
    """
    if no_match_token.upper() in response_text.upper():
        return []

    # Pattern: "TURN 45: explanation text" (captures until next TURN or end)
    turn_pattern = r'TURN\s+(\d+):\s*(.+?)(?=TURN\s+\d+:|$)'
    matches = re.findall(turn_pattern, response_text, re.DOTALL | re.IGNORECASE)

    turns = []
    for turn_num_str, explanation in matches:
        turn_num = int(turn_num_str)
        # Validate turn is within window bounds
        if window_start <= turn_num <= window_end:
            turns.append({
                'turn_number': turn_num,
                'explanation': explanation.strip()
            })

    return turns


def deduplicate_turns(turns: List[Dict]) -> List[Dict]:
    """
    Remove duplicate turn identifications (same turn_number).

    Keeps the first occurrence of each turn number.

    Args:
        turns: List of dicts with 'turn_number' and 'explanation'

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    for turn in turns:
        if turn['turn_number'] not in seen:
            seen.add(turn['turn_number'])
            result.append(turn)
    return result


def batch_consecutive_turns(turns: List[Dict], max_gap: int = 1) -> List[Dict]:
    """
    Batch consecutive turns into episodes.

    Example with max_gap=1:
        turns 5, 6, 7, 9, 10 -> episodes [(5-7), (9-10)]

    Args:
        turns: List of dicts with 'turn_number' and 'explanation'
        max_gap: Maximum gap between turns to still consider consecutive.
                 gap=1 means adjacent turns (5,6,7) form one episode.
                 gap=2 would allow one missing turn (5,7) to still be one episode.

    Returns:
        List of episode dicts with:
        - start_turn: First turn number
        - end_turn: Last turn number
        - turn_count: Number of turns in episode
        - turn_explanations: List of (turn_number, explanation) tuples
        - description: First turn's explanation (for compatibility)
        - participants: Empty list (to be filled by caller if needed)
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t['turn_number'])
    episodes = []

    current_episode = {
        'start_turn': sorted_turns[0]['turn_number'],
        'end_turn': sorted_turns[0]['turn_number'],
        'turn_explanations': [(sorted_turns[0]['turn_number'], sorted_turns[0]['explanation'])]
    }

    for turn in sorted_turns[1:]:
        if turn['turn_number'] <= current_episode['end_turn'] + max_gap:
            # Extend current episode
            current_episode['end_turn'] = turn['turn_number']
            current_episode['turn_explanations'].append(
                (turn['turn_number'], turn['explanation'])
            )
        else:
            # Finalize current episode and start new one
            _finalize_episode(current_episode)
            episodes.append(current_episode)

            current_episode = {
                'start_turn': turn['turn_number'],
                'end_turn': turn['turn_number'],
                'turn_explanations': [(turn['turn_number'], turn['explanation'])]
            }

    # Don't forget last episode
    _finalize_episode(current_episode)
    episodes.append(current_episode)

    return episodes


def _finalize_episode(episode: Dict) -> None:
    """Add computed fields to an episode dict (modifies in place)."""
    episode['turn_count'] = episode['end_turn'] - episode['start_turn'] + 1
    episode['description'] = episode['turn_explanations'][0][1]
    episode['participants'] = []  # To be filled by caller if needed


def extract_participants_from_df(
    df: pd.DataFrame,
    start_turn: int,
    end_turn: int
) -> List[str]:
    """
    Extract unique participant names from a range of turns.

    Args:
        df: Campaign DataFrame with 'character' or 'player' columns
        start_turn: Start index
        end_turn: End index (inclusive)

    Returns:
        List of unique participant names
    """
    participants = set()
    for idx in range(start_turn, min(end_turn + 1, len(df))):
        row = df.iloc[idx]
        if pd.notna(row.get('character')):
            participants.add(row['character'])
        elif pd.notna(row.get('player')):
            participants.add(row['player'])
    return list(participants)
