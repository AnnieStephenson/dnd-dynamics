"""
LLM Humor Analysis for D&D Campaigns

This module uses a 2-step LLM approach to extract and rate humor episodes
in campaign transcripts:
1. Per-turn identification: Which turns contain humor?
2. Rate each episode for originality (1-5)

Episodes are formed by batching consecutive humor turns.
Also identifies recurring/inside jokes and tracks their occurrences.
Results are aggregated by fixed-size chunks for time-series analysis.
"""

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    turn_count: int  # end_turn - start_turn + 1
    episode_text: str  # Formatted text for human review
    turn_explanations: List[tuple]  # List of (turn_number, explanation) tuples
    joke_id: Optional[str]  # For linking recurrent jokes
    is_recurring: bool  # Whether this references an earlier joke
    recurring_reference: Optional[str]  # Description of original joke if recurring


EXTRACTION_PROMPT = '''You are analyzing the transcript of a Dungeons & Dragons campaign played on an online forum for humor and jokes.

Carefully consider each turn in the transcript. For each turn that contains humor, provide:
- The turn number
- A one-sentence explanation of what makes it funny

Include:
- Standalone jokes or witty comments
- Reactions to jokes (laughter/appreciation responses)
- Collaborative humor (players building on each other's jokes)
- In-character comedic moments
- Puns, wordplay, or clever references

Do NOT include:
- Routine friendly greetings
- Unintentional humor or mistakes
- Sarcasm meant as criticism

For RECURRING JOKES (inside jokes that reference earlier jokes or running gags), mark them with [RECURRING: brief description of original joke].

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no humor found, respond with: NO_HUMOR_FOUND

Otherwise, list each turn containing humor:

TURN 45: Player makes a pun about the dragon's name
TURN 46: Other player builds on the pun with wordplay
TURN 52: [RECURRING: dragon pun] Callback to the dragon pun from earlier
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


def parse_humor_turn_extraction(
    response_text: str,
    window_start: int,
    window_end: int
) -> List[Dict]:
    """
    Parse LLM response for per-turn humor identification.

    Handles both regular turns and recurring joke markers:
    TURN 45: explanation
    TURN 52: [RECURRING: original joke] explanation

    Returns list of dicts with:
    - turn_number
    - explanation
    - is_recurring (bool)
    - recurring_reference (str or None)
    """
    if 'NO_HUMOR_FOUND' in response_text.upper():
        return []

    # Pattern: "TURN N: [RECURRING: ...] explanation" or "TURN N: explanation"
    turn_pattern = r'TURN\s+(\d+):\s*(.+?)(?=TURN\s+\d+:|$)'
    matches = re.findall(turn_pattern, response_text, re.DOTALL | re.IGNORECASE)

    turns = []
    for turn_num_str, content in matches:
        turn_num = int(turn_num_str)
        if not (window_start <= turn_num <= window_end):
            continue

        content = content.strip()

        # Check for recurring marker
        recurring_match = re.match(r'\[RECURRING:\s*([^\]]+)\]\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if recurring_match:
            recurring_reference = recurring_match.group(1).strip()
            explanation = recurring_match.group(2).strip()
            is_recurring = True
        else:
            recurring_reference = None
            explanation = content
            is_recurring = False

        turns.append({
            'turn_number': turn_num,
            'explanation': explanation,
            'is_recurring': is_recurring,
            'recurring_reference': recurring_reference
        })

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
    Preserves recurring joke info from any turn in the batch.

    Returns episodes with:
    - start_turn, end_turn
    - turn_count
    - turn_explanations: List of (turn_number, explanation)
    - description: First turn's explanation
    - is_recurring: True if any turn is recurring
    - recurring_reference: From the first recurring turn
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t['turn_number'])
    episodes = []

    current_episode = {
        'start_turn': sorted_turns[0]['turn_number'],
        'end_turn': sorted_turns[0]['turn_number'],
        'turn_explanations': [(sorted_turns[0]['turn_number'], sorted_turns[0]['explanation'])],
        'is_recurring': sorted_turns[0].get('is_recurring', False),
        'recurring_reference': sorted_turns[0].get('recurring_reference')
    }

    for turn in sorted_turns[1:]:
        if turn['turn_number'] <= current_episode['end_turn'] + max_gap:
            # Extend current episode
            current_episode['end_turn'] = turn['turn_number']
            current_episode['turn_explanations'].append(
                (turn['turn_number'], turn['explanation'])
            )
            # If any turn is recurring, mark the episode as recurring
            if turn.get('is_recurring') and not current_episode['is_recurring']:
                current_episode['is_recurring'] = True
                current_episode['recurring_reference'] = turn.get('recurring_reference')
        else:
            # Finalize current episode and start new one
            _finalize_humor_episode(current_episode)
            episodes.append(current_episode)

            current_episode = {
                'start_turn': turn['turn_number'],
                'end_turn': turn['turn_number'],
                'turn_explanations': [(turn['turn_number'], turn['explanation'])],
                'is_recurring': turn.get('is_recurring', False),
                'recurring_reference': turn.get('recurring_reference')
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


def link_inside_jokes(episodes: List[HumorEpisode]) -> Tuple[List[HumorEpisode], List[Dict]]:
    """
    Link recurring jokes by assigning consistent joke_ids.

    Returns:
        - Updated episodes with joke_ids assigned
        - List of inside joke summaries
    """
    # Find all recurring episodes
    recurring_eps = [ep for ep in episodes if ep.is_recurring and ep.recurring_reference]

    if not recurring_eps:
        return episodes, []

    # Group by similar recurring_reference (simple string matching for now)
    joke_groups = {}  # reference -> list of episodes

    for ep in recurring_eps:
        ref = ep.recurring_reference.lower().strip()
        # Find if this matches an existing group
        matched = False
        for existing_ref in joke_groups:
            # Simple substring matching - could be improved with embedding similarity
            if ref in existing_ref or existing_ref in ref:
                joke_groups[existing_ref].append(ep)
                matched = True
                break
        if not matched:
            joke_groups[ref] = [ep]

    # Assign joke_ids and build summaries
    inside_jokes = []
    joke_counter = 1

    for ref, eps in joke_groups.items():
        joke_id = f"inside_joke_{joke_counter}"
        for ep in eps:
            ep.joke_id = joke_id

        inside_jokes.append({
            'joke_id': joke_id,
            'description': ref,
            'occurrence_count': len(eps),
            'turn_indices': [(ep.start_turn, ep.end_turn) for ep in eps]
        })
        joke_counter += 1

    return episodes, inside_jokes


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
    3. Batch consecutive turns into episodes
    4. Rate each episode
    5. Link inside jokes
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

    # Step 3: Batch consecutive turns into episodes
    episode_dicts = batch_humor_turns(deduped_turns, max_gap=1)

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
            originality = rate_humor_episode(df, ep_dict, model)
            episode_text = _shared.format_turns(df, ep_dict['start_turn'], ep_dict['end_turn'])

            episodes.append(HumorEpisode(
                start_turn=ep_dict['start_turn'],
                end_turn=ep_dict['end_turn'],
                description=ep_dict['description'],
                participants=ep_dict['participants'],
                originality=originality,
                turn_count=ep_dict['turn_count'],
                episode_text=episode_text,
                turn_explanations=ep_dict.get('turn_explanations', []),
                joke_id=None,
                is_recurring=ep_dict.get('is_recurring', False),
                recurring_reference=ep_dict.get('recurring_reference')
            ))

    # Step 5: Link inside jokes
    episodes, inside_jokes = link_inside_jokes(episodes)

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
    inside_joke_occurrences = sum(ij['occurrence_count'] for ij in inside_jokes)

    summary = {
        'total_humor_episodes': len(episodes),
        'mean_humor_proportion': float(np.mean(series['humor_proportion'])) if len(series['humor_proportion']) > 0 else 0.0,
        'mean_originality': float(np.mean(all_originalities)) if all_originalities else 0.0,
        'inside_joke_count': len(inside_jokes),
        'inside_joke_occurrences': inside_joke_occurrences,
    }

    # Build metadata with episode data for human review
    episodes_data = [asdict(ep) for ep in episodes]

    print(f"  Found {len(episodes)} humor episode(s) in {len(chunks)} chunk(s), {len(inside_jokes)} inside joke(s)")

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
            'inside_jokes': inside_jokes,
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

    Uses a 2-step LLM approach:
    1. Per-turn identification of humor
    2. Rate each episode for originality (1-5)

    Episodes are formed by batching consecutive humor turns.
    Also identifies recurring/inside jokes.
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
        cache_dir = str(repo_root / 'data' / 'processed' / 'humor_results_v2')

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
