"""
LLM Judge Creativity Analysis for D&D Campaigns

This module uses an LLM as a judge to score campaign text on four creativity
dimensions: Novelty, Value, Adherence, and Resonance. Based on Sathya et al. (2026).
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import litellm

from dnd_dynamics import config
from dnd_dynamics.api_config import validate_api_key_for_model
from dnd_dynamics.llm_scaffolding.prompt_caching import retry_llm_call
from . import _cache
from .result import MetricResult


DIMENSIONS = ['novelty', 'value', 'adherence', 'resonance']

JUDGE_PROMPT = '''You are evaluating excerpts from tabletop roleplaying game (Dungeons & Dragons) sessions for creativity.

Rate the following excerpt on four dimensions of creativity (adapted from Sathya et al., 2026), providing both an overall assessment AND individual ratings for each specified character.

## Dimensions

**Novelty** (1-5): How surprising, unexpected, or original are the contributions?
- 1: Completely predictable, standard genre conventions
- 3: Some unexpected elements or choices
- 5: Genuinely surprising, never seen this before

**Value** (1-5): How clever, effective, or skillful are the contributions?
- 1: Generic or ineffective, doesn't advance the game meaningfully
- 3: Competent, some clever moments
- 5: Impressively skillful, elegant solutions

**Adherence** (1-5): How well do contributions fit the characters, world, and genre?
- 1: Breaks character, ignores established world details, feels out of place
- 3: Generally consistent, minor inconsistencies
- 5: Perfectly in character, respects world/genre, contributions feel earned

**Resonance** (1-5): How memorable or emotionally engaging is the excerpt?
- 1: Forgettable, no emotional impact
- 3: Some engaging moments
- 5: Highly memorable, would stick with you

Use the full 1-5 range. Don't hesitate to give 1s and 5s when warranted.

## Characters to Evaluate

{characters}

## Excerpt

"""
{text}
"""

## Response Format

### Overall Assessment
Novelty: [1-5]
Value: [1-5]
Adherence: [1-5]
Resonance: [1-5]
Justification: [1-2 sentences referencing specific elements]

### Character Ratings
Provide ratings for each character listed above. If a character has no contributions in this excerpt, rate all dimensions as N/A.

{character_name_1}:
- Novelty: [1-5 or N/A]
- Value: [1-5 or N/A]
- Adherence: [1-5 or N/A]
- Resonance: [1-5 or N/A]
- Note: [Brief observation, or "No contributions in this excerpt"]

(Continue for all characters in the list above)'''


def create_judge_sessions(df: pd.DataFrame, messages_per_session: int) -> List[Dict]:
    """
    Group campaign messages into sessions for judging.

    Args:
        df: Campaign DataFrame with 'text', 'player', 'character' columns
        messages_per_session: Number of messages per session

    Returns:
        List of session dicts with 'text', 'characters', 'players', 'message_indices'
    """
    sessions = []
    current_session = {
        'texts': [],
        'characters': set(),
        'players': set(),
        'message_indices': []
    }

    for idx, row in df.iterrows():
        current_session['texts'].append(row['text'])
        if pd.notna(row.get('character')):
            current_session['characters'].add(row['character'])
        if pd.notna(row.get('player')):
            current_session['players'].add(row['player'])
        current_session['message_indices'].append(idx)

        if len(current_session['texts']) >= messages_per_session:
            sessions.append({
                'text': '\n\n'.join(current_session['texts']),
                'characters': list(current_session['characters']),
                'players': list(current_session['players']),
                'message_indices': current_session['message_indices']
            })
            current_session = {
                'texts': [],
                'characters': set(),
                'players': set(),
                'message_indices': []
            }

    # Add final partial session if it has content
    if current_session['texts']:
        sessions.append({
            'text': '\n\n'.join(current_session['texts']),
            'characters': list(current_session['characters']),
            'players': list(current_session['players']),
            'message_indices': current_session['message_indices']
        })

    return sessions


def parse_judge_response(response_text: str, characters: List[str]) -> Dict:
    """
    Parse LLM judge response to extract scores.

    Args:
        response_text: Raw LLM response
        characters: List of character names to extract scores for

    Returns:
        Dict with 'overall' scores and 'by_character' scores
    """
    result = {
        'overall': {},
        'by_character': {}
    }

    # Parse overall scores
    for dim in DIMENSIONS:
        pattern = rf'{dim}:\s*(\d)'
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            result['overall'][dim] = int(match.group(1))

    # Parse character scores
    for character in characters:
        result['by_character'][character] = {}

        # Find character section - look for "CharacterName:" followed by ratings
        char_pattern = rf'{re.escape(character)}:\s*\n(.*?)(?=\n\w+:|### |$)'
        char_match = re.search(char_pattern, response_text, re.IGNORECASE | re.DOTALL)

        if char_match:
            char_section = char_match.group(1)

            for dim in DIMENSIONS:
                # Look for "- Novelty: 4" or "- Novelty: N/A"
                dim_pattern = rf'-\s*{dim}:\s*(\d|N/A)'
                dim_match = re.search(dim_pattern, char_section, re.IGNORECASE)

                if dim_match:
                    value = dim_match.group(1)
                    if value.upper() == 'N/A':
                        result['by_character'][character][dim] = None
                    else:
                        result['by_character'][character][dim] = int(value)

    return result


def judge_session(session_text: str, characters: List[str], model: str = None) -> Dict:
    """
    Call LLM to judge a single session.

    Args:
        session_text: Concatenated text of messages in the session
        characters: List of character names in this session
        model: LLM model to use (defaults to config.JUDGE_MODEL)

    Returns:
        Parsed judge response with overall and per-character scores
    """
    if model is None:
        model = config.JUDGE_MODEL

    validate_api_key_for_model(model)

    # Format character list for prompt
    char_list = "\n".join(f"- {c}" for c in characters) if characters else "- (No named characters)"
    first_char = characters[0] if characters else "Character"

    prompt = JUDGE_PROMPT.format(
        characters=char_list,
        text=session_text,
        character_name_1=first_char
    )

    response = retry_llm_call(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=config.JUDGE_TEMPERATURE
    )

    return parse_judge_response(response.choices[0].message.content, characters)


def _analyze_single_campaign_llm_judge(
    df: pd.DataFrame,
    campaign_id: str,
    messages_per_session: int = None,
    model: str = None
) -> MetricResult:
    """
    Run LLM judge analysis for a single campaign.

    Args:
        df: Campaign DataFrame
        campaign_id: Campaign identifier
        messages_per_session: Messages per judge session
        model: LLM model to use

    Returns:
        MetricResult with creativity scores
    """
    if messages_per_session is None:
        messages_per_session = config.MESSAGES_PER_JUDGE_SESSION
    if model is None:
        model = config.JUDGE_MODEL

    # Create sessions for judging
    sessions = create_judge_sessions(df, messages_per_session)

    # Build character -> player mapping
    char_to_player = {}
    for _, row in df.iterrows():
        if pd.notna(row.get('character')) and pd.notna(row.get('player')):
            char_to_player[row['character']] = row['player']

    # Collect scores
    overall_scores = {dim: [] for dim in DIMENSIONS}
    player_scores = {}  # player -> dim -> list of scores

    # Always show progress for LLM judge (API calls are slow)
    iterator = tqdm(
        sessions,
        desc=f"  {campaign_id}",
        unit="session",
        leave=True
    )

    for session in iterator:
        result = judge_session(session['text'], session['characters'], model)

        # Collect overall scores
        for dim in DIMENSIONS:
            if dim in result['overall']:
                overall_scores[dim].append(result['overall'][dim])

        # Collect per-character scores and map to players
        for character, char_scores in result['by_character'].items():
            player = char_to_player.get(character)
            if player is None:
                continue

            if player not in player_scores:
                player_scores[player] = {dim: [] for dim in DIMENSIONS}

            for dim in DIMENSIONS:
                if dim in char_scores and char_scores[dim] is not None:
                    player_scores[player][dim].append(char_scores[dim])

    # Build series arrays
    series = {dim: np.array(overall_scores[dim]) for dim in DIMENSIONS}

    # Compute composite creativity score (average of all dimensions per session)
    n_sessions = len(overall_scores[DIMENSIONS[0]]) if overall_scores[DIMENSIONS[0]] else 0
    creativity_scores = []
    for i in range(n_sessions):
        session_scores = [overall_scores[dim][i] for dim in DIMENSIONS if i < len(overall_scores[dim])]
        if session_scores:
            creativity_scores.append(np.mean(session_scores))
    series['creativity'] = np.array(creativity_scores)

    # Build summary
    summary = {}
    for dim in DIMENSIONS:
        if overall_scores[dim]:
            summary[f'mean_{dim}'] = float(np.mean(overall_scores[dim]))
        else:
            summary[f'mean_{dim}'] = np.nan
    # Add composite creativity summary
    if creativity_scores:
        summary['mean_creativity'] = float(np.mean(creativity_scores))
    else:
        summary['mean_creativity'] = np.nan

    # Build by_player
    by_player = {}
    for player, scores in player_scores.items():
        player_series = {dim: np.array(scores[dim]) for dim in DIMENSIONS}

        # Compute per-player composite creativity
        n_player_sessions = len(scores[DIMENSIONS[0]]) if scores[DIMENSIONS[0]] else 0
        player_creativity = []
        for i in range(n_player_sessions):
            session_scores = [scores[dim][i] for dim in DIMENSIONS if i < len(scores[dim])]
            if session_scores:
                player_creativity.append(np.mean(session_scores))
        player_series['creativity'] = np.array(player_creativity)

        player_summary = {}
        for dim in DIMENSIONS:
            if scores[dim]:
                player_summary[f'mean_{dim}'] = float(np.mean(scores[dim]))
            else:
                player_summary[f'mean_{dim}'] = np.nan
        if player_creativity:
            player_summary['mean_creativity'] = float(np.mean(player_creativity))
        else:
            player_summary['mean_creativity'] = np.nan

        by_player[player] = MetricResult(
            series=player_series,
            summary=player_summary
        )

    return MetricResult(
        series=series,
        summary=summary,
        by_player=by_player,
        metadata={
            'campaign_id': campaign_id,
            'judge_model': model,
            'messages_per_session': messages_per_session,
            'session_count': len(sessions),
            'total_messages': len(df),
        }
    )


def analyze_llm_judge_creativity(
    data: Dict[str, pd.DataFrame],
    messages_per_session: int = None,
    model: str = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, MetricResult]:
    """
    Calculate LLM judge creativity metrics for campaigns.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        messages_per_session: Messages per judge session (default: config.MESSAGES_PER_JUDGE_SESSION)
        model: LLM model for judging (default: config.JUDGE_MODEL)
        show_progress: Whether to show progress bars
        cache_dir: Directory for caching (default: data/processed/llm_judge_creativity_results)
        force_refresh: Force recomputation ignoring cache

    Returns:
        Dict[campaign_id, MetricResult]
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")

    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'llm_judge_creativity_results')

    # Handle caching
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "LLM judge creativity"
    )

    # Process missing campaigns
    new_results = {}
    if data_to_process:
        total_campaigns = len(data_to_process)
        if show_progress:
            print(f"Judging {total_campaigns} campaign(s)...")

        for i, (campaign_id, df) in enumerate(data_to_process.items(), 1):
            if show_progress:
                print(f"Campaign {i}/{total_campaigns}: {campaign_id}")
            new_results[campaign_id] = _analyze_single_campaign_llm_judge(
                df, campaign_id,
                messages_per_session=messages_per_session,
                model=model
            )

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "LLM judge creativity"
    )
