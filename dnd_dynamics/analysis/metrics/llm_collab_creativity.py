"""
LLM Collaborative Creativity Analysis for D&D Campaigns

This module uses a 2-step LLM approach to extract and rate collaborative creativity
episodes in campaign transcripts:
1. Extract episodes where players build on each other's narrative contributions
2. Rate each episode for creativity (1-5)

Includes: world-building, character development, shared storytelling, creative problem-solving.
Excludes: purely tactical/mechanical coordination.

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
from .result import MetricResult


@dataclass
class CollabCreativityEpisode:
    start_turn: int  # DataFrame index
    end_turn: int    # DataFrame index
    description: str
    participants: List[str]
    creativity: int  # 1-5 rating
    turn_count: int  # end_turn - start_turn + 1
    episode_text: str  # Formatted text for human review


EXTRACTION_PROMPT = '''You are analyzing a tabletop roleplaying game transcript for moments of collaborative creativity.

Identify episodes where players build on each other's narrative contributions. Include:
- **World-building**: Players adding details to the setting, creating lore together
- **Character development**: Meaningful character interactions that develop relationships or backstories
- **Shared storytelling**: Players contributing to and building on each other's narrative ideas
- **Creative problem-solving**: Unconventional or imaginative solutions developed collaboratively

This may include collaborative humor, but also non-humorous creative building.

Do NOT include:
- Purely tactical or mechanical coordination (e.g., "I'll flank left, you flank right")
- Combat mechanics without narrative creativity (e.g., "I'll cast bless, then you attack")
- Simple back-and-forth dialogue without creative expansion
- One player's solo creative moment (must involve building on each other)

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no collaborative creativity found, respond with: NO_COLLAB_CREATIVITY_FOUND

Otherwise, list each episode:

EPISODE 1:
Start turn: [number]
End turn: [number]
Description: [What creative collaboration occurred]
Participants: [comma-separated list]

EPISODE 2:
...
'''


RATING_PROMPT = '''Rate this collaborative creativity episode from a tabletop roleplaying game.

## Episode (turns {start_turn} to {end_turn})

{episode_text}

## Description
{description}

## Rating

**Creativity** (1-5):
- 1: Minimal creativity - predictable, standard contributions
- 2: Some creativity - a few unexpected elements
- 3: Moderately creative - interesting ideas that build on each other
- 4: Highly creative - surprising, clever collaborative building
- 5: Exceptionally creative - genuinely original, memorable collaboration

Consider: How novel/surprising? How skillfully executed? How well does it fit the world/characters?

IMPORTANT: Rate the quality of ideas, not the quantity of words. A brief, clever exchange can be highly creative. A long, elaborate passage may be uncreative if the ideas are predictable.

Use the full 1-5 range.

## Response Format
Creativity: [1-5]
Reasoning: [1 sentence]
'''


def format_turns_for_collab_creativity(df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
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

    Returns list of dicts with: start_turn, end_turn, description, participants
    """
    if 'NO_COLLAB_CREATIVITY_FOUND' in response_text.upper():
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
        part_match = re.search(r'Participants:\s*(.+?)(?=\n|$)', match, re.IGNORECASE)
        if part_match:
            participants_str = part_match.group(1).strip()
            episode['participants'] = [p.strip() for p in participants_str.split(',')]

        # Validate episode has required fields and is within window bounds
        if all(k in episode for k in ['start_turn', 'end_turn', 'description', 'participants']):
            # Clamp to window bounds
            episode['start_turn'] = max(episode['start_turn'], window_start)
            episode['end_turn'] = min(episode['end_turn'], window_end)
            if episode['start_turn'] <= episode['end_turn']:
                episodes.append(episode)

    return episodes


def extract_collab_creativity_from_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    model: str
) -> List[Dict]:
    """Extract collaborative creativity episodes from a window of turns."""
    validate_api_key_for_model(model)

    text = format_turns_for_collab_creativity(df, start_idx, end_idx)
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


def rate_collab_creativity_episode(
    df: pd.DataFrame,
    episode: Dict,
    model: str
) -> int:
    """Rate a single collaborative creativity episode (1-5)."""
    validate_api_key_for_model(model)

    episode_text = format_turns_for_collab_creativity(df, episode['start_turn'], episode['end_turn'])
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

    # Parse creativity from response
    response_text = response.choices[0].message.content
    creativity_match = re.search(r'Creativity:\s*(\d)', response_text, re.IGNORECASE)
    if creativity_match:
        return int(creativity_match.group(1))
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


def create_collab_creativity_chunks(
    total_turns: int,
    episodes: List[CollabCreativityEpisode],
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
    episodes: List[CollabCreativityEpisode]
) -> Dict:
    """Calculate metrics for a single chunk."""
    chunk_turns = chunk_end - chunk_start

    # Find episodes that overlap with this chunk
    overlapping = []
    total_collab_creativity_turns = 0

    for ep in episodes:
        # Check overlap
        if ep.start_turn < chunk_end and ep.end_turn >= chunk_start:
            overlapping.append(ep)
            # Count turns within this chunk
            overlap_start = max(ep.start_turn, chunk_start)
            overlap_end = min(ep.end_turn + 1, chunk_end)
            total_collab_creativity_turns += overlap_end - overlap_start

    collab_creativity_episodes = len(overlapping)
    collab_creativity_proportion = total_collab_creativity_turns / chunk_turns if chunk_turns > 0 else 0
    mean_creativity = np.mean([ep.creativity for ep in overlapping]) if overlapping else 0

    return {
        'collab_creativity_episodes': collab_creativity_episodes,
        'collab_creativity_turns': total_collab_creativity_turns,
        'collab_creativity_proportion': collab_creativity_proportion,
        'mean_creativity': mean_creativity
    }


def _analyze_single_campaign_collab_creativity(
    df: pd.DataFrame,
    campaign_id: str,
    chunk_size: int = None,
    model: str = None
) -> MetricResult:
    """
    Full collaborative creativity analysis pipeline for a single campaign.

    1. Extract episodes using sliding windows (if needed)
    2. Deduplicate overlapping episode detections
    3. Rate each episode
    4. Create chunks with adjusted boundaries
    5. Aggregate per-chunk metrics
    6. Build MetricResult
    """
    if chunk_size is None:
        chunk_size = config.SOCIAL_CHUNK_SIZE
    if model is None:
        model = config.SOCIAL_MODEL

    total_turns = len(df)
    window_size = config.SOCIAL_EXTRACTION_WINDOW
    overlap = config.SOCIAL_EXTRACTION_OVERLAP

    # Step 1: Extract using sliding windows
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
        window_episodes = extract_collab_creativity_from_window(df, start_idx, end_idx, model)
        all_raw_episodes.extend(window_episodes)

    # Step 2: Deduplicate
    deduped_episodes = deduplicate_episodes(all_raw_episodes)

    # Step 3: Rate each episode
    episodes = []
    if deduped_episodes:
        print(f"  Rating {len(deduped_episodes)} episode(s)...")
        for ep_dict in tqdm(deduped_episodes, desc=f"  {campaign_id} rating", unit="episode"):
            creativity = rate_collab_creativity_episode(df, ep_dict, model)
            episode_text = format_turns_for_collab_creativity(df, ep_dict['start_turn'], ep_dict['end_turn'])

            episodes.append(CollabCreativityEpisode(
                start_turn=ep_dict['start_turn'],
                end_turn=ep_dict['end_turn'],
                description=ep_dict['description'],
                participants=ep_dict['participants'],
                creativity=creativity,
                turn_count=ep_dict['end_turn'] - ep_dict['start_turn'] + 1,
                episode_text=episode_text
            ))

    # Step 4: Create chunks with adjusted boundaries
    chunks = create_collab_creativity_chunks(total_turns, episodes, chunk_size)

    # Step 5: Aggregate per-chunk metrics
    chunk_metrics = [aggregate_chunk_metrics(c['start'], c['end'], episodes) for c in chunks]

    # Build series arrays
    series = {
        'collab_creativity_episodes': np.array([m['collab_creativity_episodes'] for m in chunk_metrics]),
        'collab_creativity_turns': np.array([m['collab_creativity_turns'] for m in chunk_metrics]),
        'collab_creativity_proportion': np.array([m['collab_creativity_proportion'] for m in chunk_metrics]),
        'mean_creativity': np.array([m['mean_creativity'] for m in chunk_metrics]),
    }

    # Build summary
    all_creativities = [ep.creativity for ep in episodes]
    summary = {
        'total_collab_creativity_episodes': len(episodes),
        'mean_collab_creativity_proportion': float(np.mean(series['collab_creativity_proportion'])) if len(series['collab_creativity_proportion']) > 0 else 0.0,
        'mean_creativity': float(np.mean(all_creativities)) if all_creativities else 0.0,
    }

    # Build metadata with episode data for human review
    episodes_data = [asdict(ep) for ep in episodes]

    print(f"  Found {len(episodes)} collab creativity episode(s) in {len(chunks)} chunk(s)")

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


def analyze_collab_creativity(
    data: Dict[str, pd.DataFrame],
    chunk_size: int = None,
    model: str = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, MetricResult]:
    """
    Analyze collaborative creativity in campaign transcripts.

    Uses a 2-step LLM approach:
    1. Extract collaborative creativity episodes from text
    2. Rate each episode for creativity (1-5)

    Results are aggregated by fixed-size chunks.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        chunk_size: Turns per chunk (default: config.SOCIAL_CHUNK_SIZE)
        model: LLM model (default: config.SOCIAL_MODEL)
        show_progress: Whether to show progress bars
        cache_dir: Directory for caching (default: data/processed/collab_creativity_results)
        force_refresh: Force recomputation ignoring cache

    Returns:
        Dict[campaign_id, MetricResult]
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")

    if cache_dir is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        cache_dir = str(repo_root / 'data' / 'processed' / 'collab_creativity_results')

    # Handle caching
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "Collab creativity"
    )

    # Process missing campaigns
    new_results = {}
    if data_to_process:
        total_campaigns = len(data_to_process)
        if show_progress:
            print(f"Analyzing collaborative creativity in {total_campaigns} campaign(s)...")

        for i, (campaign_id, df) in enumerate(data_to_process.items(), 1):
            if show_progress:
                print(f"Campaign {i}/{total_campaigns}: {campaign_id} ({len(df)} turns)")
            new_results[campaign_id] = _analyze_single_campaign_collab_creativity(
                df, campaign_id,
                chunk_size=chunk_size,
                model=model
            )

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "Collab creativity"
    )
