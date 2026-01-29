"""
LLM Norms Analysis for D&D Campaigns

This module uses an LLM to extract social norm episodes from campaign transcripts:
- Norm establishment (explicit agreements or implicit patterns)
- Norm following (adhering to established norms)
- Norm violation (breaking established norms)
- Norm enforcement (reacting when someone violates a norm)

Tracks recurring norms across the campaign with per-norm statistics.
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
from .result import MetricResult


EPISODE_TYPES = ['establishing', 'following', 'violating', 'enforcing']


@dataclass
class NormEpisode:
    start_turn: int  # DataFrame index
    end_turn: int    # DataFrame index
    description: str  # Description of the norm itself
    episode_description: str  # What happened in this episode
    participants: List[str]
    episode_type: str  # 'establishing', 'following', 'violating', 'enforcing'
    is_explicit: bool  # Explicit rule vs implicit pattern
    norm_id: Optional[str]  # Links episodes about the same norm
    turn_count: int  # end_turn - start_turn + 1
    episode_text: str  # Formatted text for human review


EXTRACTION_PROMPT = '''You are analyzing a tabletop roleplaying game transcript for social norms.

Identify norm-related episodes - moments where group norms are established, followed, violated, or enforced.

Look for:
- **Establishing**: Creating a new norm (explicit agreement or implicit pattern emerging)
  - Explicit: "Let's always try diplomacy first", "We vote on major decisions"
  - Implicit: Repeated pattern like "the rogue always scouts ahead"
- **Following**: Adhering to an established norm
- **Violating**: Breaking an established norm
- **Enforcing**: Reacting when someone violates a norm ("Hey, we agreed to share loot!")

Types of norms:
- Decision-making processes (voting, consulting specific members)
- Resource sharing (loot distribution, spell slots)
- Role expectations (who leads, who scouts, who negotiates)
- Play style agreements (diplomacy first, no stealing from party)
- Character interaction rules (respecting backstories, no PvP)

Do NOT include:
- In-character roleplay/banter, even if characters react negatively to each other. 
  A norm reflects how PLAYERS expect the game to be played, not how CHARACTERS treat each other in-fiction. 
  If it's just characters being grumpy/teasing/rude to each other and players are having fun, it's not a norm.
- Game mechanics or rules (these are DM-enforced, not group norms)
- One-time decisions that don't become patterns
- Combat tactics (unless they're explicit group agreements)

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no norm-related episodes found, respond with: NO_NORMS_FOUND

Otherwise, list each episode:

EPISODE 1:
Start turn: [number]
End turn: [number]
Type: [establishing/following/violating/enforcing]
Explicit: [YES/NO]
Norm: [Brief description of the norm itself]
Description: [What happened in this episode]
Participants: [comma-separated list]

EPISODE 2:
...
'''


def format_turns_for_norms(df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
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

    Returns list of dicts with: start_turn, end_turn, episode_type, is_explicit,
    description (norm), episode_description, participants
    """
    if 'NO_NORMS_FOUND' in response_text.upper():
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

        # Parse type
        type_match = re.search(r'Type:\s*(establishing|following|violating|enforcing)', match, re.IGNORECASE)
        if type_match:
            episode['episode_type'] = type_match.group(1).lower()

        # Parse explicit
        explicit_match = re.search(r'Explicit:\s*(YES|NO)', match, re.IGNORECASE)
        if explicit_match:
            episode['is_explicit'] = explicit_match.group(1).upper() == 'YES'
        else:
            episode['is_explicit'] = False

        # Parse norm description
        norm_match = re.search(r'Norm:\s*(.+?)(?=\n|Description:|$)', match, re.IGNORECASE | re.DOTALL)
        if norm_match:
            episode['description'] = norm_match.group(1).strip()

        # Parse episode description
        desc_match = re.search(r'Description:\s*(.+?)(?=\n|Participants:|$)', match, re.IGNORECASE | re.DOTALL)
        if desc_match:
            episode['episode_description'] = desc_match.group(1).strip()

        # Parse participants
        part_match = re.search(r'Participants:\s*(.+?)(?=\n|$)', match, re.IGNORECASE)
        if part_match:
            participants_str = part_match.group(1).strip()
            episode['participants'] = [p.strip() for p in participants_str.split(',')]

        # Validate episode has required fields and is within window bounds
        required = ['start_turn', 'end_turn', 'episode_type', 'description', 'participants']
        if all(k in episode for k in required) and episode['episode_type'] in EPISODE_TYPES:
            # Clamp to window bounds
            episode['start_turn'] = max(episode['start_turn'], window_start)
            episode['end_turn'] = min(episode['end_turn'], window_end)
            if episode['start_turn'] <= episode['end_turn']:
                # Set default episode_description if missing
                if 'episode_description' not in episode:
                    episode['episode_description'] = episode['description']
                episodes.append(episode)

    return episodes


def extract_norms_from_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    model: str
) -> List[Dict]:
    """Extract norm episodes from a window of turns."""
    validate_api_key_for_model(model)

    text = format_turns_for_norms(df, start_idx, end_idx)
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


def link_norms(episodes: List[NormEpisode]) -> Tuple[List[NormEpisode], List[Dict]]:
    """
    Link episodes about the same norm by assigning consistent norm_ids.
    Group by similar norm descriptions.

    Returns:
        - Updated episodes with norm_ids
        - List of norm summaries with per-norm stats
    """
    if not episodes:
        return episodes, []

    # Group episodes by similar norm description (simple substring matching)
    norm_groups = {}  # description -> list of episodes

    for ep in episodes:
        desc = ep.description.lower().strip()
        # Find if this matches an existing group
        matched = False
        for existing_desc in list(norm_groups.keys()):
            # Simple substring matching - could be improved with embedding similarity
            if desc in existing_desc or existing_desc in desc or _similar_descriptions(desc, existing_desc):
                norm_groups[existing_desc].append(ep)
                matched = True
                break
        if not matched:
            norm_groups[desc] = [ep]

    # Assign norm_ids and build summaries
    norms = []
    norm_counter = 1

    for desc, eps in norm_groups.items():
        norm_id = f"norm_{norm_counter}"
        for ep in eps:
            ep.norm_id = norm_id

        # Count by episode type
        times_established = sum(1 for ep in eps if ep.episode_type == 'establishing')
        times_followed = sum(1 for ep in eps if ep.episode_type == 'following')
        times_violated = sum(1 for ep in eps if ep.episode_type == 'violating')
        times_enforced = sum(1 for ep in eps if ep.episode_type == 'enforcing')

        # Determine if norm is explicit (if any episode marks it as explicit)
        is_explicit = any(ep.is_explicit for ep in eps)

        # Use the longest description as the canonical one
        canonical_desc = max([ep.description for ep in eps], key=len)

        norms.append({
            'norm_id': norm_id,
            'description': canonical_desc,
            'is_explicit': is_explicit,
            'times_established': times_established,
            'times_followed': times_followed,
            'times_violated': times_violated,
            'times_enforced': times_enforced,
            'total_episodes': len(eps),
            'episode_indices': [(ep.start_turn, ep.end_turn) for ep in eps]
        })
        norm_counter += 1

    return episodes, norms


def _similar_descriptions(desc1: str, desc2: str) -> bool:
    """Check if two norm descriptions are similar using word overlap."""
    words1 = set(desc1.lower().split())
    words2 = set(desc2.lower().split())
    # Remove common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for'}
    words1 -= stopwords
    words2 -= stopwords
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    min_size = min(len(words1), len(words2))
    return overlap >= min_size * 0.5  # 50% word overlap


def create_norm_chunks(
    total_turns: int,
    episodes: List[NormEpisode],
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
    episodes: List[NormEpisode]
) -> Dict:
    """Calculate metrics for a single chunk."""
    chunk_turns = chunk_end - chunk_start

    # Find episodes that overlap with this chunk
    overlapping = []
    total_norm_turns = 0

    for ep in episodes:
        # Check overlap
        if ep.start_turn < chunk_end and ep.end_turn >= chunk_start:
            overlapping.append(ep)
            # Count turns within this chunk
            overlap_start = max(ep.start_turn, chunk_start)
            overlap_end = min(ep.end_turn + 1, chunk_end)
            total_norm_turns += overlap_end - overlap_start

    norm_episodes = len(overlapping)
    norm_proportion = total_norm_turns / chunk_turns if chunk_turns > 0 else 0

    return {
        'norm_episodes': norm_episodes,
        'norm_turns': total_norm_turns,
        'norm_proportion': norm_proportion
    }


def _analyze_single_campaign_norms(
    df: pd.DataFrame,
    campaign_id: str,
    chunk_size: int = None,
    model: str = None
) -> MetricResult:
    """
    Full norms analysis pipeline for a single campaign.

    1. Extract norms using sliding windows (if needed)
    2. Deduplicate overlapping episode detections
    3. Link norms (group episodes about same norm)
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

    # Step 1: Extract norms using sliding windows
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
        window_episodes = extract_norms_from_window(df, start_idx, end_idx, model)
        all_raw_episodes.extend(window_episodes)

    # Step 2: Deduplicate
    deduped_episodes = deduplicate_episodes(all_raw_episodes)

    # Step 3: Convert to NormEpisode objects
    episodes = []
    for ep_dict in deduped_episodes:
        episode_text = format_turns_for_norms(df, ep_dict['start_turn'], ep_dict['end_turn'])

        episodes.append(NormEpisode(
            start_turn=ep_dict['start_turn'],
            end_turn=ep_dict['end_turn'],
            description=ep_dict['description'],
            episode_description=ep_dict.get('episode_description', ep_dict['description']),
            participants=ep_dict['participants'],
            episode_type=ep_dict['episode_type'],
            is_explicit=ep_dict.get('is_explicit', False),
            norm_id=None,
            turn_count=ep_dict['end_turn'] - ep_dict['start_turn'] + 1,
            episode_text=episode_text
        ))

    # Step 4: Link norms
    episodes, norms = link_norms(episodes)

    # Step 5: Create chunks with adjusted boundaries
    chunks = create_norm_chunks(total_turns, episodes, chunk_size)

    # Step 6: Aggregate per-chunk metrics
    chunk_metrics = [aggregate_chunk_metrics(c['start'], c['end'], episodes) for c in chunks]

    # Build series arrays
    series = {
        'norm_episodes': np.array([m['norm_episodes'] for m in chunk_metrics]),
        'norm_turns': np.array([m['norm_turns'] for m in chunk_metrics]),
        'norm_proportion': np.array([m['norm_proportion'] for m in chunk_metrics]),
    }

    # Build summary
    total_established = sum(1 for ep in episodes if ep.episode_type == 'establishing')
    total_followed = sum(1 for ep in episodes if ep.episode_type == 'following')
    total_violated = sum(1 for ep in episodes if ep.episode_type == 'violating')
    total_enforced = sum(1 for ep in episodes if ep.episode_type == 'enforcing')

    summary = {
        'total_norm_episodes': len(episodes),
        'total_unique_norms': len(norms),
        'mean_norm_proportion': float(np.mean(series['norm_proportion'])) if len(series['norm_proportion']) > 0 else 0.0,
        'total_established': total_established,
        'total_followed': total_followed,
        'total_violated': total_violated,
        'total_enforced': total_enforced,
    }

    # Build metadata with episode data for human review
    episodes_data = [asdict(ep) for ep in episodes]

    print(f"  Found {len(episodes)} norm episode(s) across {len(norms)} unique norm(s) in {len(chunks)} chunk(s)")

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
            'norms': norms,
        }
    )


def analyze_norms(
    data: Dict[str, pd.DataFrame],
    chunk_size: int = None,
    model: str = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, MetricResult]:
    """
    Analyze social norms in campaign transcripts.

    Detects norm-related episodes:
    - Establishing: Creating a new norm
    - Following: Adhering to an established norm
    - Violating: Breaking an established norm
    - Enforcing: Reacting when someone violates a norm

    Results are aggregated by fixed-size chunks.

    Args:
        data: Dict of DataFrames {campaign_id: df}
        chunk_size: Turns per chunk (default: config.SOCIAL_CHUNK_SIZE)
        model: LLM model (default: config.SOCIAL_MODEL)
        show_progress: Whether to show progress bars
        cache_dir: Directory for caching (default: data/processed/norms_results)
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
        cache_dir = str(repo_root / 'data' / 'processed' / 'norms_results')

    # Handle caching (model-aware)
    cached_results, data_to_process = _cache.handle_multi_campaign_caching(
        data, cache_dir, force_refresh, show_progress, "Norms", model=model
    )

    # Process missing campaigns
    new_results = {}
    if data_to_process:
        total_campaigns = len(data_to_process)
        if show_progress:
            print(f"Analyzing norms in {total_campaigns} campaign(s)...")

        for i, (campaign_id, df) in enumerate(data_to_process.items(), 1):
            if show_progress:
                print(f"Campaign {i}/{total_campaigns}: {campaign_id} ({len(df)} turns)")
            new_results[campaign_id] = _analyze_single_campaign_norms(
                df, campaign_id,
                chunk_size=chunk_size,
                model=model
            )

    return _cache.save_new_results_and_combine(
        cached_results, new_results, cache_dir, show_progress, "Norms", model=model
    )
