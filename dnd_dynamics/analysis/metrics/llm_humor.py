"""
LLM Humor Analysis for D&D Campaigns

This module uses a 2-step LLM approach to extract and rate humor episodes
in campaign transcripts:
1. Extract humor episodes from text (with sliding windows for large campaigns)
2. Rate each episode for originality (1-5)

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
from dnd_dynamics.llm_scaffolding.prompt_caching import retry_llm_call
from . import _cache
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
    joke_id: Optional[str]  # For linking recurrent jokes
    is_recurring: bool  # Whether this references an earlier joke
    recurring_reference: Optional[str]  # Description of original joke if recurring


EXTRACTION_PROMPT = '''You are analyzing a tabletop roleplaying game transcript for humor and jokes.

Identify humor episodes - moments of intentional comedy, jokes, or playful banter. Include:
- Standalone jokes or witty comments (can be single turn)
- Jokes with reactions (joke + laughter/appreciation responses)
- Collaborative humor (players building on each other's jokes)
- In-character comedic moments
- Puns, wordplay, or clever references

Do NOT include:
- Routine friendly greetings
- Unintentional humor or mistakes
- Sarcasm meant as criticism

Also identify RECURRING JOKES (inside jokes) - humor that references earlier jokes or running gags in the campaign. Mark these with a consistent label.

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no humor found, respond with: NO_HUMOR_FOUND

Otherwise, list each humor episode:

EPISODE 1:
Start turn: [number]
End turn: [number]
Description: [Brief description of what makes it funny]
Participants: [comma-separated list]
Recurring: [NO or YES - "references <brief description of original joke>"]

EPISODE 2:
...
'''


RATING_PROMPT = '''Rate this humor episode from a tabletop roleplaying game.

## Humor Episode (turns {start_turn} to {end_turn})

{episode_text}

## Description
{description}

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


def format_turns_for_humor(df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
    """
    Format DataFrame rows as readable text with character names.

    Output format:
    Turn 45 - CharacterName: message text here...
    Turn 46 - OtherCharacter: their message text...
    """
    lines = []
    for idx in range(start_idx, min(end_idx + 1, len(df))):
        row = df.iloc[idx]
        character = row.get('character', 'Unknown')
        text = row.get('text', '')
        lines.append(f"Turn {idx} - {character}: {text}")
    return '\n'.join(lines)


def parse_extraction_response(response_text: str, window_start: int, window_end: int) -> List[Dict]:
    """
    Parse LLM extraction response to get episode dictionaries.

    Returns list of dicts with: start_turn, end_turn, description, participants, is_recurring, recurring_reference
    """
    if 'NO_HUMOR_FOUND' in response_text.upper():
        return []

    episodes = []
    episode_pattern = r'EPISODE\s+\d+:\s*\n(.*?)(?=EPISODE\s+\d+:|$)'
    matches = re.findall(episode_pattern, response_text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        episode = {}

        # Parse start turn
        start_match = re.search(r'Start turn:\s*(\d+)', match, re.IGNORECASE)
        if start_match:
            episode['start_turn'] = int(start_match.group(1))

        # Parse end turn
        end_match = re.search(r'End turn:\s*(\d+)', match, re.IGNORECASE)
        if end_match:
            episode['end_turn'] = int(end_match.group(1))

        # Parse description
        desc_match = re.search(r'Description:\s*(.+?)(?=\n|Participants:|$)', match, re.IGNORECASE | re.DOTALL)
        if desc_match:
            episode['description'] = desc_match.group(1).strip()

        # Parse participants
        part_match = re.search(r'Participants:\s*(.+?)(?=\n|Recurring:|$)', match, re.IGNORECASE)
        if part_match:
            participants_str = part_match.group(1).strip()
            episode['participants'] = [p.strip() for p in participants_str.split(',')]

        # Parse recurring
        recurring_match = re.search(r'Recurring:\s*(.+?)(?=\n|$)', match, re.IGNORECASE)
        if recurring_match:
            recurring_text = recurring_match.group(1).strip()
            if recurring_text.upper().startswith('YES'):
                episode['is_recurring'] = True
                # Extract reference description
                ref_match = re.search(r'references?\s+["\']?(.+?)["\']?\s*$', recurring_text, re.IGNORECASE)
                if ref_match:
                    episode['recurring_reference'] = ref_match.group(1).strip()
                else:
                    episode['recurring_reference'] = recurring_text[4:].strip()  # Remove "YES -" or "YES"
            else:
                episode['is_recurring'] = False
                episode['recurring_reference'] = None
        else:
            episode['is_recurring'] = False
            episode['recurring_reference'] = None

        # Validate episode has required fields and is within window bounds
        if all(k in episode for k in ['start_turn', 'end_turn', 'description', 'participants']):
            # Clamp to window bounds
            episode['start_turn'] = max(episode['start_turn'], window_start)
            episode['end_turn'] = min(episode['end_turn'], window_end)
            if episode['start_turn'] <= episode['end_turn']:
                episodes.append(episode)

    return episodes


def extract_humor_from_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    model: str
) -> List[Dict]:
    """Extract humor episodes from a window of turns."""
    validate_api_key_for_model(model)

    text = format_turns_for_humor(df, start_idx, end_idx)
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
        temperature=0.3  # Lower temp for extraction to be consistent
    )

    return parse_extraction_response(response.choices[0].message.content, start_idx, end_idx)


def rate_humor_episode(
    df: pd.DataFrame,
    episode: Dict,
    model: str
) -> int:
    """Rate a single humor episode originality (1-5)."""
    validate_api_key_for_model(model)

    episode_text = format_turns_for_humor(df, episode['start_turn'], episode['end_turn'])
    prompt = RATING_PROMPT.format(
        start_turn=episode['start_turn'],
        end_turn=episode['end_turn'],
        episode_text=episode_text,
        description=episode['description']
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


def deduplicate_episodes(episodes: List[Dict]) -> List[Dict]:
    """
    Deduplicate episodes that have >50% turn overlap.
    Keeps the episode with longer description.
    """
    if not episodes:
        return []

    # Sort by start turn
    sorted_eps = sorted(episodes, key=lambda e: e['start_turn'])
    result = []

    for ep in sorted_eps:
        ep_turns = set(range(ep['start_turn'], ep['end_turn'] + 1))
        merged = False

        for i, existing in enumerate(result):
            existing_turns = set(range(existing['start_turn'], existing['end_turn'] + 1))
            overlap = len(ep_turns & existing_turns)
            min_size = min(len(ep_turns), len(existing_turns))

            if overlap > min_size * 0.5:  # >50% overlap
                # Keep episode with longer description
                if len(ep.get('description', '')) > len(existing.get('description', '')):
                    result[i] = ep
                merged = True
                break

        if not merged:
            result.append(ep)

    return result


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

    1. Extract humor using sliding windows (if needed)
    2. Deduplicate overlapping episode detections
    3. Rate each episode
    4. Link inside jokes
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

    # Step 1: Extract humor using sliding windows
    all_raw_episodes = []

    if total_turns <= window_size:
        # Single window
        windows = [(0, total_turns - 1)]
    else:
        # Sliding windows with overlap
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
        window_episodes = extract_humor_from_window(df, start_idx, end_idx, model)
        all_raw_episodes.extend(window_episodes)

    # Step 2: Deduplicate
    deduped_episodes = deduplicate_episodes(all_raw_episodes)

    # Step 3: Rate each episode
    episodes = []
    if deduped_episodes:
        print(f"  Rating {len(deduped_episodes)} episode(s)...")
        for ep_dict in tqdm(deduped_episodes, desc=f"  {campaign_id} rating", unit="episode"):
            originality = rate_humor_episode(df, ep_dict, model)
            episode_text = format_turns_for_humor(df, ep_dict['start_turn'], ep_dict['end_turn'])

            episodes.append(HumorEpisode(
                start_turn=ep_dict['start_turn'],
                end_turn=ep_dict['end_turn'],
                description=ep_dict['description'],
                participants=ep_dict['participants'],
                originality=originality,
                turn_count=ep_dict['end_turn'] - ep_dict['start_turn'] + 1,
                episode_text=episode_text,
                joke_id=None,
                is_recurring=ep_dict.get('is_recurring', False),
                recurring_reference=ep_dict.get('recurring_reference')
            ))

    # Step 4: Link inside jokes
    episodes, inside_jokes = link_inside_jokes(episodes)

    # Step 5: Create chunks with adjusted boundaries
    chunks = create_humor_chunks(total_turns, episodes, chunk_size)

    # Step 6: Aggregate per-chunk metrics
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
    1. Extract humor episodes from text
    2. Rate each episode for originality (1-5)

    Also identifies recurring/inside jokes.
    Results are aggregated by fixed-size chunks.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        chunk_size: Turns per chunk (default: config.SOCIAL_CHUNK_SIZE)
        model: LLM model (default: config.SOCIAL_MODEL)
        show_progress: Whether to show progress bars
        cache_dir: Directory for caching (default: data/processed/humor_results)
        force_refresh: Force recomputation ignoring cache

    Returns:
        Dict[campaign_id, MetricResult]
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")

    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'humor_results')

    # Handle caching
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "Humor"
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
        cached_results, new_results, cache_dir, show_progress, "Humor"
    )
