"""
LLM Humor Analysis for D&D Campaigns

This module uses a 3-step LLM approach to extract and analyze humor episodes
in campaign transcripts:
1. Per-turn identification: Cast wide net to identify all humor turns
2. Classification: Classify each turn as ORIGIN (new joke) or CALLBACK (reference)
3. Rating: Rate each episode for originality (1-5)

Episodes are formed by batching consecutive humor turns.
Results are aggregated by fixed-size chunks for time-series analysis.
"""

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import litellm

from dnd_dynamics import config
from dnd_dynamics.api_config import validate_api_key_for_model
from dnd_dynamics.api_config import retry_llm_call
from . import _cache
from . import _shared
from .result import MetricResult


@dataclass
class HumorEpisode:
    start_turn: int  # DataFrame index
    end_turn: int    # DataFrame index
    description: str
    participants: List[str]
    originality: int  # 1-5
    humor_type: str  # 'ORIGIN' or 'CALLBACK' (majority of turns)
    turn_count: int  # end_turn - start_turn + 1
    episode_text: str  # Formatted text for human review
    turn_explanations: List[tuple]  # List of (turn_number, explanation) tuples
    turn_types: List[tuple]  # List of (turn_number, humor_type) for per-turn tracking


EXTRACTION_PROMPT = '''You are analyzing the transcript of a Dungeons & Dragons campaign played on an online forum for humor and jokes.

Evaluate EVERY turn in the transcript. Do not skip any turns. Cast a wide net to identify ALL turns that contain:
- Jokes or funny moments (even subtle ones)
- References or callbacks to earlier humor in the campaign
- Reactions to humor (laughter, appreciation, building on jokes)

For each turn that contains humor, provide:
- The turn number
- A one-sentence explanation of what makes it funny or why it's humor-related

Include:
- Standalone jokes or witty comments
- Reactions to jokes (laughter/appreciation responses)
- Collaborative humor (players building on each other's jokes)
- In-character comedic moments
- Puns, wordplay, or clever references
- References to earlier jokes (even if not funny on their own)

Do NOT include:
- Routine friendly greetings
- Unintentional humor or mistakes
- Sarcasm meant as criticism

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no humor found, respond with: NO_HUMOR_FOUND

Otherwise, list each turn containing humor:

TURN 45: Player makes a pun about the dragon's name
TURN 46: Other player builds on the pun with wordplay
TURN 52: Reference to the dragon pun from earlier
...
'''


RATING_PROMPT = '''Rate this humor episode from a Dungeons & Dragons campaign played on an online forum.

## Humor Episode (turns {start_turn} to {end_turn})

{episode_text}

## Turn-by-turn analysis:
{turn_explanations}

## Rating

**Originality** (1-5):
- 1: Very common joke type, predictable punchline
- 2: Standard humor, slightly predictable
- 3: Moderately original, some unexpected elements
- 4: Creative and unexpected
- 5: Highly original, surprising or clever twist

Use the full 1-5 range.

## Response Format
Originality: [1-5]
Reasoning: [1 sentence]
'''


CLASSIFICATION_PROMPT = '''You are classifying humor turns from a Dungeons & Dragons campaign played on an online forum.

For each turn listed below, classify it as either:
- ORIGIN: New joke in the campaign, actually funny on its own
- CALLBACK: References earlier humor from the campaign, may or may not be funny on its own

## Turns to classify:

{turn_list}

## Response Format

Classify each turn:

TURN 45: ORIGIN
TURN 46: CALLBACK
TURN 52: ORIGIN
...
'''


def parse_humor_turn_extraction(
    response_text: str,
    window_start: int,
    window_end: int
) -> List[Dict]:
    """
    Parse LLM response for per-turn humor identification.

    Returns list of dicts with:
    - turn_number
    - explanation
    """
    if 'NO_HUMOR_FOUND' in response_text.upper():
        return []

    # Pattern: "TURN N: explanation"
    turn_pattern = r'TURN\s+(\d+):\s*(.+?)(?=TURN\s+\d+:|$)'
    matches = re.findall(turn_pattern, response_text, re.DOTALL | re.IGNORECASE)

    turns = []
    for turn_num_str, content in matches:
        turn_num = int(turn_num_str)
        if not (window_start <= turn_num <= window_end):
            continue

        turns.append({
            'turn_number': turn_num,
            'explanation': content.strip()
        })

    return turns


def classify_humor_turns(
    turns: List[Dict],
    model: str
) -> List[Dict]:
    """
    Classify each humor turn as ORIGIN or CALLBACK using LLM.

    Args:
        turns: List of dicts with 'turn_number' and 'explanation'
        model: LLM model to use

    Returns:
        Updated turns with 'humor_type' added ('ORIGIN' or 'CALLBACK')
    """
    if not turns:
        return turns

    validate_api_key_for_model(model)

    # Format turns for classification prompt
    turn_list = "\n".join(
        f"TURN {t['turn_number']}: {t['explanation']}"
        for t in turns
    )

    prompt = CLASSIFICATION_PROMPT.format(turn_list=turn_list)

    response = retry_llm_call(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )

    # Parse classification response
    response_text = response.choices[0].message.content
    classifications = {}

    for line in response_text.strip().split('\n'):
        match = re.search(r'TURN\s+(\d+):\s*(ORIGIN|CALLBACK)', line, re.IGNORECASE)
        if match:
            turn_num = int(match.group(1))
            humor_type = match.group(2).upper()
            classifications[turn_num] = humor_type

    # Update turns with classifications
    for turn in turns:
        turn['humor_type'] = classifications.get(turn['turn_number'], 'ORIGIN')  # Default to ORIGIN

    return turns


def extract_humor_turns_from_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    model: str
) -> List[Dict]:
    """Extract humor turns from a window using per-turn identification."""
    validate_api_key_for_model(model)

    text = _shared.format_turns(df, start_idx, end_idx)
    prompt = EXTRACTION_PROMPT.format(
        start_turn=start_idx,
        end_turn=end_idx,
        text=text
    )

    response = retry_llm_call(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3
    )

    return parse_humor_turn_extraction(
        response.choices[0].message.content,
        start_idx,
        end_idx
    )


def rate_humor_episode(
    df: pd.DataFrame,
    episode: Dict,
    model: str
) -> int:
    """Rate a single humor episode originality (1-5)."""
    validate_api_key_for_model(model)

    episode_text = _shared.format_turns(df, episode['start_turn'], episode['end_turn'])

    # Format turn explanations for the prompt
    turn_explanations_str = "\n".join(
        f"Turn {turn_num}: {explanation}"
        for turn_num, explanation in episode.get('turn_explanations', [])
    )

    prompt = RATING_PROMPT.format(
        start_turn=episode['start_turn'],
        end_turn=episode['end_turn'],
        episode_text=episode_text,
        turn_explanations=turn_explanations_str or episode.get('description', '')
    )

    response = retry_llm_call(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3
    )

    # Parse originality from response
    response_text = response.choices[0].message.content
    originality_match = re.search(r'Originality:\s*(\d)', response_text, re.IGNORECASE)
    if originality_match:
        return int(originality_match.group(1))
    return 3  # Default to middle if parsing fails


def deduplicate_humor_turns(turns: List[Dict]) -> List[Dict]:
    """
    Remove duplicate turn identifications (same turn_number).
    Keeps the first occurrence of each turn number.
    Preserves recurring info.
    """
    seen = set()
    result = []
    for turn in turns:
        if turn['turn_number'] not in seen:
            seen.add(turn['turn_number'])
            result.append(turn)
    return result


def batch_humor_turns(turns: List[Dict], max_gap: int = 1) -> List[Dict]:
    """
    Batch consecutive humor turns into episodes.
    Tracks humor_type (ORIGIN/CALLBACK) for each turn.

    Returns episodes with:
    - start_turn, end_turn
    - turn_count
    - turn_explanations: List of (turn_number, explanation)
    - turn_types: List of (turn_number, humor_type)
    - description: First turn's explanation
    - humor_type: Majority type among turns (ORIGIN if tied)
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t['turn_number'])
    episodes = []

    current_episode = {
        'start_turn': sorted_turns[0]['turn_number'],
        'end_turn': sorted_turns[0]['turn_number'],
        'turn_explanations': [(sorted_turns[0]['turn_number'], sorted_turns[0]['explanation'])],
        'turn_types': [(sorted_turns[0]['turn_number'], sorted_turns[0].get('humor_type', 'ORIGIN'))]
    }

    for turn in sorted_turns[1:]:
        if turn['turn_number'] <= current_episode['end_turn'] + max_gap:
            # Extend current episode
            current_episode['end_turn'] = turn['turn_number']
            current_episode['turn_explanations'].append(
                (turn['turn_number'], turn['explanation'])
            )
            current_episode['turn_types'].append(
                (turn['turn_number'], turn.get('humor_type', 'ORIGIN'))
            )
        else:
            # Finalize current episode and start new one
            _finalize_humor_episode(current_episode)
            episodes.append(current_episode)

            current_episode = {
                'start_turn': turn['turn_number'],
                'end_turn': turn['turn_number'],
                'turn_explanations': [(turn['turn_number'], turn['explanation'])],
                'turn_types': [(turn['turn_number'], turn.get('humor_type', 'ORIGIN'))]
            }

    # Don't forget last episode
    _finalize_humor_episode(current_episode)
    episodes.append(current_episode)

    return episodes


def _finalize_humor_episode(episode: Dict) -> None:
    """Add computed fields to an episode dict (modifies in place)."""
    episode['turn_count'] = episode['end_turn'] - episode['start_turn'] + 1
    episode['description'] = episode['turn_explanations'][0][1]
    episode['participants'] = []  # To be filled by caller

    # Determine majority humor_type
    origin_count = sum(1 for _, t in episode['turn_types'] if t == 'ORIGIN')
    callback_count = sum(1 for _, t in episode['turn_types'] if t == 'CALLBACK')
    episode['humor_type'] = 'CALLBACK' if callback_count > origin_count else 'ORIGIN'


def create_humor_chunks(
    total_turns: int,
    episodes: List[HumorEpisode],
    chunk_size: int
) -> List[Dict]:
    """
    Create chunks with adjusted boundaries to avoid splitting episodes.

    Returns list of dicts with 'start' and 'end' (exclusive) indices.
    """
    if total_turns == 0:
        return []

    chunks = []
    current_start = 0

    while current_start < total_turns:
        natural_end = min(current_start + chunk_size, total_turns)

        # Check if any episode spans the boundary
        adjusted_end = natural_end
        for ep in episodes:
            # Episode crosses boundary if it starts before and ends at/after
            if ep.start_turn < natural_end <= ep.end_turn:
                adjusted_end = max(adjusted_end, ep.end_turn + 1)

        adjusted_end = min(adjusted_end, total_turns)  # Don't exceed total

        chunks.append({
            'start': current_start,
            'end': adjusted_end
        })

        current_start = adjusted_end

    return chunks


def aggregate_chunk_metrics(
    chunk_start: int,
    chunk_end: int,
    episodes: List[HumorEpisode]
) -> Dict:
    """Calculate metrics for a single chunk."""
    chunk_turns = chunk_end - chunk_start

    # Find episodes that overlap with this chunk
    overlapping = []
    total_humor_turns = 0

    for ep in episodes:
        # Check overlap
        if ep.start_turn < chunk_end and ep.end_turn >= chunk_start:
            overlapping.append(ep)
            # Count turns within this chunk
            overlap_start = max(ep.start_turn, chunk_start)
            overlap_end = min(ep.end_turn + 1, chunk_end)
            total_humor_turns += overlap_end - overlap_start

    humor_episodes = len(overlapping)
    humor_proportion = total_humor_turns / chunk_turns if chunk_turns > 0 else 0
    mean_originality = np.mean([ep.originality for ep in overlapping]) if overlapping else 0

    return {
        'humor_episodes': humor_episodes,
        'humor_turns': total_humor_turns,
        'humor_proportion': humor_proportion,
        'mean_originality': mean_originality
    }


def _analyze_single_campaign_humor(
    df: pd.DataFrame,
    campaign_id: str,
    chunk_size: int = None,
    model: str = None
) -> MetricResult:
    """
    Full humor analysis pipeline for a single campaign.

    1. Extract humor turns using sliding windows
    2. Deduplicate turns
    3. Classify each turn as ORIGIN or CALLBACK
    4. Batch consecutive turns into episodes
    5. Rate each episode for originality
    6. Create chunks with adjusted boundaries
    7. Aggregate per-chunk metrics
    8. Build MetricResult
    """
    if chunk_size is None:
        chunk_size = config.SOCIAL_CHUNK_SIZE
    if model is None:
        model = config.SOCIAL_MODEL

    total_turns = len(df)
    window_size = config.SOCIAL_EXTRACTION_WINDOW
    overlap = config.SOCIAL_EXTRACTION_OVERLAP

    # Step 1: Extract humor turns using sliding windows
    all_turns = []

    if total_turns <= window_size:
        windows = [(0, total_turns - 1)]
    else:
        windows = []
        start = 0
        while start < total_turns:
            end = min(start + window_size - 1, total_turns - 1)
            windows.append((start, end))
            start += window_size - overlap
            if start >= total_turns:
                break

    print(f"  Extracting from {len(windows)} window(s)...")
    for start_idx, end_idx in tqdm(windows, desc=f"  {campaign_id} extraction", unit="window"):
        window_turns = extract_humor_turns_from_window(df, start_idx, end_idx, model)
        all_turns.extend(window_turns)

    # Step 2: Deduplicate turns
    deduped_turns = deduplicate_humor_turns(all_turns)

    # Step 3: Classify each turn as ORIGIN or CALLBACK
    if deduped_turns:
        print(f"  Classifying {len(deduped_turns)} turn(s)...")
        deduped_turns = classify_humor_turns(deduped_turns, model)

    # Step 4: Batch consecutive turns into episodes
    episode_dicts = batch_humor_turns(deduped_turns, max_gap=1)

    # Add participants to each episode
    for ep in episode_dicts:
        ep['participants'] = _shared.extract_participants_from_df(
            df, ep['start_turn'], ep['end_turn']
        )

    # Step 5: Rate each episode for originality
    episodes = []
    if episode_dicts:
        print(f"  Rating {len(episode_dicts)} episode(s)...")
        for ep_dict in tqdm(episode_dicts, desc=f"  {campaign_id} rating", unit="episode"):
            originality = rate_humor_episode(df, ep_dict, model)
            episode_text = _shared.format_turns(df, ep_dict['start_turn'], ep_dict['end_turn'])

            episodes.append(HumorEpisode(
                start_turn=ep_dict['start_turn'],
                end_turn=ep_dict['end_turn'],
                description=ep_dict['description'],
                participants=ep_dict['participants'],
                originality=originality,
                humor_type=ep_dict.get('humor_type', 'ORIGIN'),
                turn_count=ep_dict['turn_count'],
                episode_text=episode_text,
                turn_explanations=ep_dict.get('turn_explanations', []),
                turn_types=ep_dict.get('turn_types', [])
            ))

    # Step 6: Create chunks with adjusted boundaries
    chunks = create_humor_chunks(total_turns, episodes, chunk_size)

    # Step 7: Aggregate per-chunk metrics
    chunk_metrics = [aggregate_chunk_metrics(c['start'], c['end'], episodes) for c in chunks]

    # Build series arrays
    series = {
        'humor_episodes': np.array([m['humor_episodes'] for m in chunk_metrics]),
        'humor_turns': np.array([m['humor_turns'] for m in chunk_metrics]),
        'humor_proportion': np.array([m['humor_proportion'] for m in chunk_metrics]),
        'mean_originality': np.array([m['mean_originality'] for m in chunk_metrics]),
    }

    # Build summary
    all_originalities = [ep.originality for ep in episodes]
    origin_episodes = [ep for ep in episodes if ep.humor_type == 'ORIGIN']
    callback_episodes = [ep for ep in episodes if ep.humor_type == 'CALLBACK']

    summary = {
        'total_humor_episodes': len(episodes),
        'origin_count': len(origin_episodes),
        'callback_count': len(callback_episodes),
        'mean_humor_proportion': float(np.mean(series['humor_proportion'])) if len(series['humor_proportion']) > 0 else 0.0,
        'mean_originality': float(np.mean(all_originalities)) if all_originalities else 0.0,
    }

    # Build metadata with episode data for human review
    episodes_data = [asdict(ep) for ep in episodes]

    print(f"  Found {len(episodes)} humor episode(s) in {len(chunks)} chunk(s) ({len(origin_episodes)} origin, {len(callback_episodes)} callback)")

    return MetricResult(
        series=series,
        summary=summary,
        by_player={},
        metadata={
            'campaign_id': campaign_id,
            'model': model,
            'chunk_size': chunk_size,
            'total_chunks': len(chunks),
            'total_turns': total_turns,
            'episodes': episodes_data,
        }
    )


def analyze_humor(
    data: Dict[str, pd.DataFrame],
    chunk_size: int = None,
    model: str = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, MetricResult]:
    """
    Analyze humor and jokes in campaign transcripts.

    Uses a 3-step LLM approach:
    1. Per-turn identification of humor (wide net)
    2. Classification of each turn as ORIGIN or CALLBACK
    3. Rate each episode for originality (1-5)

    Episodes are formed by batching consecutive humor turns.
    Results are aggregated by fixed-size chunks.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        chunk_size: Turns per chunk (default: config.SOCIAL_CHUNK_SIZE)
        model: LLM model (default: config.SOCIAL_MODEL)
        show_progress: Whether to show progress bars
        cache_dir: Directory for caching (default: data/processed/humor_results_v2)
        force_refresh: Force recomputation ignoring cache

    Returns:
        Dict[campaign_id, MetricResult]
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")

    if model is None:
        model = config.SOCIAL_MODEL
    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'humor_results_v3')

    # Handle caching (model-aware)
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "Humor", model=model
    )

    # Process missing campaigns
    new_results = {}
    if data_to_process:
        total_campaigns = len(data_to_process)
        if show_progress:
            print(f"Analyzing humor in {total_campaigns} campaign(s)...")

        for i, (campaign_id, df) in enumerate(data_to_process.items(), 1):
            if show_progress:
                print(f"Campaign {i}/{total_campaigns}: {campaign_id} ({len(df)} turns)")
            new_results[campaign_id] = _analyze_single_campaign_humor(
                df, campaign_id,
                chunk_size=chunk_size,
                model=model
            )

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "Humor", model=model
    )
