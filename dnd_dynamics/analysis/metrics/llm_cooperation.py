"""
LLM Cooperation Analysis for D&D Campaigns

This module uses a 2-step LLM approach to extract and rate cooperative moments
in campaign transcripts:
1. Per-turn identification: Which turns contain cooperation?
2. Rate each episode for depth/substance (1-5)

Episodes are formed by batching consecutive cooperation turns.
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
class CooperationEpisode:
    start_turn: int  # DataFrame index
    end_turn: int    # DataFrame index
    description: str
    participants: List[str]
    depth: int  # 1-5 (substance rating)
    cooperation_type: str  # 'ACTUAL' or 'LANGUAGE'
    turn_count: int  # end_turn - start_turn + 1
    episode_text: str  # Formatted text for human review
    turn_explanations: List[tuple]  # List of (turn_number, explanation) tuples


EXTRACTION_PROMPT = '''You are analyzing the transcript of a Dungeons & Dragons campaign played on an online forum for cooperative moments between players/characters.

Evaluate EVERY turn in the transcript. Do not skip any turns. For each turn that contains cooperation, provide:
- The turn number
- A type marker: [ACTUAL] or [LANGUAGE]
- A one-sentence explanation of the cooperation

Cooperation types:
- ACTUAL: Characters directly coordinate action - enabling, supporting, or building on another's actions
- LANGUAGE: Characters making speeches about teamwork/unity without actual coordinated action

Include:
- One character helping or aiding another
- Characters coordinating to solve a problem together
- Collective decision-making or group planning
- Compromises or negotiations that reach agreement
- Collaborative strategizing or teamwork
- Speeches or statements about unity and working together (mark as LANGUAGE)

Do NOT include:
- Combat coordination against NPCs/monsters (unless unusually collaborative)
- Routine turn-taking or game mechanics
- One character simply following another's orders without input

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no cooperation found, respond with: NO_COOPERATION_FOUND

Otherwise, list each turn containing cooperation with its type:

TURN 45 [ACTUAL]: Characters discuss strategy for entering the dungeon
TURN 46 [ACTUAL]: Party agrees on formation and backup plan
TURN 52 [LANGUAGE]: Character gives inspiring speech about teamwork
...
'''


def parse_cooperation_turns_with_type(
    response_text: str,
    start_idx: int,
    end_idx: int
) -> List[Dict]:
    """Parse cooperation turns with type markers from LLM response."""
    if "NO_COOPERATION_FOUND" in response_text:
        return []

    turns = []
    # Match: TURN 45 [ACTUAL]: explanation or TURN 45 [LANGUAGE]: explanation
    pattern = r'TURN\s+(\d+)\s*\[(\w+)\]\s*:\s*(.+)'

    for line in response_text.strip().split('\n'):
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            turn_num = int(match.group(1))
            cooperation_type = match.group(2).upper()
            explanation = match.group(3).strip()

            # Validate turn is in range
            if start_idx <= turn_num <= end_idx:
                # Normalize type to ACTUAL or LANGUAGE
                if cooperation_type not in ('ACTUAL', 'LANGUAGE'):
                    cooperation_type = 'ACTUAL'  # Default

                turns.append({
                    'turn_number': turn_num,
                    'explanation': explanation,
                    'cooperation_type': cooperation_type
                })

    return turns


def determine_episode_cooperation_type(turns: List[Dict], episode_dict: Dict) -> str:
    """
    Determine the cooperation type for an episode based on its constituent turns.

    Uses majority vote among turns in the episode. Ties default to ACTUAL.
    """
    episode_turns = [
        t for t in turns
        if episode_dict['start_turn'] <= t['turn_number'] <= episode_dict['end_turn']
    ]

    if not episode_turns:
        return 'ACTUAL'  # Default

    actual_count = sum(1 for t in episode_turns if t.get('cooperation_type') == 'ACTUAL')
    language_count = sum(1 for t in episode_turns if t.get('cooperation_type') == 'LANGUAGE')

    return 'LANGUAGE' if language_count > actual_count else 'ACTUAL'


RATING_PROMPT = '''Rate this cooperation episode from a Dungeons & Dragons campaign played on an online forum.

## Cooperation Episode (turns {start_turn} to {end_turn})

{episode_text}

## Turn-by-turn analysis:
{turn_explanations}

## Rating

**Depth** (1-5):
- 1: Superficial - brief acknowledgment or minimal assistance
- 2: Light - simple help or quick agreement
- 3: Moderate - meaningful coordination or discussion
- 4: Substantial - significant collaboration with back-and-forth
- 5: Deep - extensive teamwork, complex problem-solving together

Use the full 1-5 range.

## Response Format
Depth: [1-5]
Reasoning: [1 sentence]
'''


def extract_cooperation_turns_from_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    model: str
) -> List[Dict]:
    """Extract cooperation turns from a window using per-turn identification."""
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

    return parse_cooperation_turns_with_type(
        response.choices[0].message.content,
        start_idx,
        end_idx
    )


def rate_cooperation_episode(
    df: pd.DataFrame,
    episode: Dict,
    model: str
) -> int:
    """Rate a single cooperation episode depth (1-5)."""
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

    # Parse depth from response
    response_text = response.choices[0].message.content
    depth_match = re.search(r'Depth:\s*(\d)', response_text, re.IGNORECASE)
    if depth_match:
        return int(depth_match.group(1))
    return 3  # Default to middle if parsing fails


def create_cooperation_chunks(
    total_turns: int,
    episodes: List[CooperationEpisode],
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
    episodes: List[CooperationEpisode]
) -> Dict:
    """Calculate metrics for a single chunk."""
    chunk_turns = chunk_end - chunk_start

    # Find episodes that overlap with this chunk
    overlapping = []
    total_cooperation_turns = 0

    for ep in episodes:
        # Check overlap
        if ep.start_turn < chunk_end and ep.end_turn >= chunk_start:
            overlapping.append(ep)
            # Count turns within this chunk
            overlap_start = max(ep.start_turn, chunk_start)
            overlap_end = min(ep.end_turn + 1, chunk_end)
            total_cooperation_turns += overlap_end - overlap_start

    cooperation_episodes = len(overlapping)
    cooperation_proportion = total_cooperation_turns / chunk_turns if chunk_turns > 0 else 0
    mean_depth = np.mean([ep.depth for ep in overlapping]) if overlapping else 0

    return {
        'cooperation_episodes': cooperation_episodes,
        'cooperation_turns': total_cooperation_turns,
        'cooperation_proportion': cooperation_proportion,
        'mean_depth': mean_depth
    }


def _analyze_single_campaign_cooperation(
    df: pd.DataFrame,
    campaign_id: str,
    chunk_size: int = None,
    model: str = None
) -> MetricResult:
    """
    Full cooperation analysis pipeline for a single campaign.

    1. Extract cooperation turns using sliding windows
    2. Deduplicate turns
    3. Batch consecutive turns into episodes
    4. Rate each episode
    5. Create chunks with adjusted boundaries
    6. Aggregate per-chunk metrics
    7. Build MetricResult
    """
    if chunk_size is None:
        chunk_size = config.SOCIAL_CHUNK_SIZE
    if model is None:
        model = config.SOCIAL_MODEL

    total_turns = len(df)
    window_size = config.SOCIAL_EXTRACTION_WINDOW
    overlap = config.SOCIAL_EXTRACTION_OVERLAP

    # Step 1: Extract cooperation turns using sliding windows
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
        window_turns = extract_cooperation_turns_from_window(df, start_idx, end_idx, model)
        all_turns.extend(window_turns)

    # Step 2: Deduplicate turns
    deduped_turns = _shared.deduplicate_turns(all_turns)

    # Step 3: Batch consecutive turns into episodes
    episode_dicts = _shared.batch_consecutive_turns(deduped_turns, max_gap=1)

    # Add participants to each episode
    for ep in episode_dicts:
        ep['participants'] = _shared.extract_participants_from_df(
            df, ep['start_turn'], ep['end_turn']
        )

    # Step 4: Rate each episode
    episodes = []
    if episode_dicts:
        print(f"  Rating {len(episode_dicts)} episode(s)...")
        for ep_dict in tqdm(episode_dicts, desc=f"  {campaign_id} rating", unit="episode"):
            depth = rate_cooperation_episode(df, ep_dict, model)
            episode_text = _shared.format_turns(df, ep_dict['start_turn'], ep_dict['end_turn'])
            cooperation_type = determine_episode_cooperation_type(deduped_turns, ep_dict)

            episodes.append(CooperationEpisode(
                start_turn=ep_dict['start_turn'],
                end_turn=ep_dict['end_turn'],
                description=ep_dict['description'],
                participants=ep_dict['participants'],
                depth=depth,
                cooperation_type=cooperation_type,
                turn_count=ep_dict['turn_count'],
                episode_text=episode_text,
                turn_explanations=ep_dict.get('turn_explanations', [])
            ))

    # Step 5: Create chunks with adjusted boundaries
    chunks = create_cooperation_chunks(total_turns, episodes, chunk_size)

    # Step 6: Aggregate per-chunk metrics
    chunk_metrics = [aggregate_chunk_metrics(c['start'], c['end'], episodes) for c in chunks]

    # Build series arrays
    series = {
        'cooperation_episodes': np.array([m['cooperation_episodes'] for m in chunk_metrics]),
        'cooperation_turns': np.array([m['cooperation_turns'] for m in chunk_metrics]),
        'cooperation_proportion': np.array([m['cooperation_proportion'] for m in chunk_metrics]),
        'mean_depth': np.array([m['mean_depth'] for m in chunk_metrics]),
    }

    # Build summary
    all_depths = [ep.depth for ep in episodes]
    actual_episodes = [ep for ep in episodes if ep.cooperation_type == 'ACTUAL']
    language_episodes = [ep for ep in episodes if ep.cooperation_type == 'LANGUAGE']
    summary = {
        'total_cooperation_episodes': len(episodes),
        'actual_cooperation_count': len(actual_episodes),
        'language_cooperation_count': len(language_episodes),
        'mean_cooperation_proportion': float(np.mean(series['cooperation_proportion'])) if len(series['cooperation_proportion']) > 0 else 0.0,
        'mean_depth': float(np.mean(all_depths)) if all_depths else 0.0,
    }

    # Build metadata with episode data for human review
    episodes_data = [asdict(ep) for ep in episodes]

    print(f"  Found {len(episodes)} cooperation episode(s) in {len(chunks)} chunk(s)")

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


def analyze_cooperation(
    data: Dict[str, pd.DataFrame],
    chunk_size: int = None,
    model: str = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, MetricResult]:
    """
    Analyze cooperative moments in campaign transcripts.

    Uses a 2-step LLM approach:
    1. Per-turn identification of cooperation
    2. Rate each episode for depth (1-5)

    Episodes are formed by batching consecutive cooperation turns.
    Results are aggregated by fixed-size chunks.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        chunk_size: Turns per chunk (default: config.SOCIAL_CHUNK_SIZE)
        model: LLM model (default: config.SOCIAL_MODEL)
        show_progress: Whether to show progress bars
        cache_dir: Directory for caching (default: data/processed/cooperation_results_v2)
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
        cache_dir = str(repo_root / 'data' / 'processed' / 'cooperation_results_v3')

    # Handle caching (model-aware)
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "Cooperation", model=model
    )

    # Process missing campaigns
    new_results = {}
    if data_to_process:
        total_campaigns = len(data_to_process)
        if show_progress:
            print(f"Analyzing cooperation in {total_campaigns} campaign(s)...")

        for i, (campaign_id, df) in enumerate(data_to_process.items(), 1):
            if show_progress:
                print(f"Campaign {i}/{total_campaigns}: {campaign_id} ({len(df)} turns)")
            new_results[campaign_id] = _analyze_single_campaign_cooperation(
                df, campaign_id,
                chunk_size=chunk_size,
                model=model
            )

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "Cooperation", model=model
    )
