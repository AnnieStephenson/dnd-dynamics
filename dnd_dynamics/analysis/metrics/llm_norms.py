"""
LLM Norms Analysis for D&D Campaigns

This module uses a 2-step LLM approach to extract and categorize social norm episodes
in campaign transcripts:
1. Per-turn identification: Which turns contain norm-related activity?
2. Categorize each episode (type, explicit/implicit, norm description)

Episode types:
- Norm establishment (explicit agreements or implicit patterns)
- Norm following (adhering to established norms)
- Norm violation (breaking established norms)
- Norm enforcement (reacting when someone violates a norm)

Episodes are formed by batching consecutive norm-related turns.
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
from . import _shared
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
    turn_explanations: List[tuple]  # List of (turn_number, explanation) tuples


EXTRACTION_PROMPT = '''You are analyzing the transcript of a Dungeons & Dragons campaign played on an online forum for social norms.

Evaluate EVERY turn in the transcript. Do not skip any turns. For each turn that contains norm-related activity, provide:
- The turn number
- The type of norm activity: [establishing], [following], [violating], or [enforcing]
- A one-sentence explanation

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
- Characters making speeches about unity, purpose, or shared values
- Combat coordination (calling out tactics mid-fight, attacking the same enemy)
- Characters encouraging each other during challenges
- Standard D&D gameplay any party would do (scouting, healing allies, sharing loot)
- In-character roleplay or banter between characters
- Game mechanics or rules (these are DM-enforced, not group norms)
- One-time decisions that don't become patterns
- One-time planning or coordination, even if explicit ("let's do X tonight") - must be a recurring pattern

## Transcript (turns {start_turn} to {end_turn})

{text}

## Response Format

If no norm-related activity found, respond with: NO_NORMS_FOUND

Otherwise, list each turn containing norm-related activity:

TURN 45 [establishing]: Player proposes that the party always votes on major decisions
TURN 46 [following]: Party members agree to the voting proposal
TURN 52 [violating]: Player makes a unilateral decision without consulting the party
TURN 53 [enforcing]: Another player calls out the violation of the voting norm
...
'''


CATEGORIZATION_PROMPT = '''Categorize this social norm episode from a Dungeons & Dragons campaign played on an online forum.

## Episode (turns {start_turn} to {end_turn})

{episode_text}

## Turn-by-turn analysis:
{turn_explanations}

## Categorization

Based on the episode, provide:

1. **Episode Type** (the PRIMARY activity in this episode):
   - establishing: A new norm is being created or proposed
   - following: An existing norm is being adhered to
   - violating: An established norm is being broken
   - enforcing: Someone is calling out a norm violation

2. **Explicit**: Is this an EXPLICIT norm (stated rule/agreement) or IMPLICIT (pattern/expectation)?
   - YES: The norm is explicitly stated ("we agreed to...", "let's always...")
   - NO: The norm is implicit (unspoken expectation, emerging pattern)

3. **Norm Description**: A brief description of the norm itself (not the episode)
   - e.g., "Party votes on major decisions", "Rogue scouts ahead", "Share loot equally"

4. **Episode Description**: What happened in this specific episode

## Response Format
Type: [establishing/following/violating/enforcing]
Explicit: [YES/NO]
Norm: [Brief description of the norm]
Description: [What happened in this episode]
'''


def parse_norm_turn_extraction(
    response_text: str,
    window_start: int,
    window_end: int
) -> List[Dict]:
    """
    Parse LLM response for per-turn norm identification.

    Expects format:
    TURN 45 [establishing]: explanation
    TURN 46 [following]: explanation

    Returns list of dicts with:
    - turn_number
    - explanation
    - type_hint (establishing/following/violating/enforcing)
    """
    if 'NO_NORMS_FOUND' in response_text.upper():
        return []

    # Pattern: "TURN N [type]: explanation"
    turn_pattern = r'TURN\s+(\d+)\s*\[(establishing|following|violating|enforcing)\]:\s*(.+?)(?=TURN\s+\d+\s*\[|$)'
    matches = re.findall(turn_pattern, response_text, re.DOTALL | re.IGNORECASE)

    turns = []
    for turn_num_str, type_hint, explanation in matches:
        turn_num = int(turn_num_str)
        if window_start <= turn_num <= window_end:
            turns.append({
                'turn_number': turn_num,
                'explanation': explanation.strip(),
                'type_hint': type_hint.lower()
            })

    return turns


def extract_norm_turns_from_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    model: str
) -> List[Dict]:
    """Extract norm-related turns from a window using per-turn identification."""
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

    return parse_norm_turn_extraction(
        response.choices[0].message.content,
        start_idx,
        end_idx
    )


def categorize_norm_episode(
    df: pd.DataFrame,
    episode: Dict,
    model: str
) -> Dict:
    """
    Categorize a single norm episode.

    Returns dict with:
    - episode_type
    - is_explicit
    - description (norm)
    - episode_description
    """
    validate_api_key_for_model(model)

    episode_text = _shared.format_turns(df, episode['start_turn'], episode['end_turn'])

    # Format turn explanations for the prompt
    turn_explanations_str = "\n".join(
        f"Turn {turn_num} [{episode.get('type_hints', {}).get(turn_num, 'unknown')}]: {explanation}"
        for turn_num, explanation in episode.get('turn_explanations', [])
    )

    prompt = CATEGORIZATION_PROMPT.format(
        start_turn=episode['start_turn'],
        end_turn=episode['end_turn'],
        episode_text=episode_text,
        turn_explanations=turn_explanations_str
    )

    response = retry_llm_call(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3
    )

    response_text = response.choices[0].message.content

    # Parse episode type
    type_match = re.search(r'Type:\s*(establishing|following|violating|enforcing)', response_text, re.IGNORECASE)
    episode_type = type_match.group(1).lower() if type_match else _infer_type_from_hints(episode)

    # Parse explicit
    explicit_match = re.search(r'Explicit:\s*(YES|NO)', response_text, re.IGNORECASE)
    is_explicit = explicit_match.group(1).upper() == 'YES' if explicit_match else False

    # Parse norm description
    norm_match = re.search(r'Norm:\s*(.+?)(?=\n|Description:|$)', response_text, re.IGNORECASE | re.DOTALL)
    norm_description = norm_match.group(1).strip() if norm_match else episode.get('description', '')

    # Parse episode description
    desc_match = re.search(r'Description:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE | re.DOTALL)
    episode_description = desc_match.group(1).strip() if desc_match else episode.get('description', '')

    return {
        'episode_type': episode_type,
        'is_explicit': is_explicit,
        'description': norm_description,
        'episode_description': episode_description
    }


def _infer_type_from_hints(episode: Dict) -> str:
    """Infer episode type from type hints if categorization parsing fails."""
    type_hints = episode.get('type_hints', {})
    if not type_hints:
        return 'establishing'  # Default

    # Use most common type hint
    from collections import Counter
    counts = Counter(type_hints.values())
    return counts.most_common(1)[0][0]


def batch_norm_turns(turns: List[Dict], max_gap: int = 1) -> List[Dict]:
    """
    Batch consecutive norm turns into episodes.
    Preserves type hints from each turn.

    Returns episodes with:
    - start_turn, end_turn
    - turn_count
    - turn_explanations: List of (turn_number, explanation)
    - type_hints: Dict mapping turn_number to type_hint
    - description: First turn's explanation
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t['turn_number'])
    episodes = []

    current_episode = {
        'start_turn': sorted_turns[0]['turn_number'],
        'end_turn': sorted_turns[0]['turn_number'],
        'turn_explanations': [(sorted_turns[0]['turn_number'], sorted_turns[0]['explanation'])],
        'type_hints': {sorted_turns[0]['turn_number']: sorted_turns[0].get('type_hint', 'unknown')}
    }

    for turn in sorted_turns[1:]:
        if turn['turn_number'] <= current_episode['end_turn'] + max_gap:
            # Extend current episode
            current_episode['end_turn'] = turn['turn_number']
            current_episode['turn_explanations'].append(
                (turn['turn_number'], turn['explanation'])
            )
            current_episode['type_hints'][turn['turn_number']] = turn.get('type_hint', 'unknown')
        else:
            # Finalize current episode and start new one
            _finalize_norm_episode(current_episode)
            episodes.append(current_episode)

            current_episode = {
                'start_turn': turn['turn_number'],
                'end_turn': turn['turn_number'],
                'turn_explanations': [(turn['turn_number'], turn['explanation'])],
                'type_hints': {turn['turn_number']: turn.get('type_hint', 'unknown')}
            }

    # Don't forget last episode
    _finalize_norm_episode(current_episode)
    episodes.append(current_episode)

    return episodes


def _finalize_norm_episode(episode: Dict) -> None:
    """Add computed fields to an episode dict (modifies in place)."""
    episode['turn_count'] = episode['end_turn'] - episode['start_turn'] + 1
    episode['description'] = episode['turn_explanations'][0][1]
    episode['participants'] = []  # To be filled by caller


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

    1. Extract norm turns using sliding windows
    2. Deduplicate turns
    3. Batch consecutive turns into episodes
    4. Categorize each episode (type, explicit, norm description)
    5. Link norms (group episodes about same norm)
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

    # Step 1: Extract norm turns using sliding windows
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
        window_turns = extract_norm_turns_from_window(df, start_idx, end_idx, model)
        all_turns.extend(window_turns)

    # Step 2: Deduplicate turns
    deduped_turns = _shared.deduplicate_turns(all_turns)

    # Step 3: Batch consecutive turns into episodes
    episode_dicts = batch_norm_turns(deduped_turns, max_gap=1)

    # Add participants to each episode
    for ep in episode_dicts:
        ep['participants'] = _shared.extract_participants_from_df(
            df, ep['start_turn'], ep['end_turn']
        )

    # Step 4: Categorize each episode
    episodes = []
    if episode_dicts:
        print(f"  Categorizing {len(episode_dicts)} episode(s)...")
        for ep_dict in tqdm(episode_dicts, desc=f"  {campaign_id} categorizing", unit="episode"):
            categorization = categorize_norm_episode(df, ep_dict, model)
            episode_text = _shared.format_turns(df, ep_dict['start_turn'], ep_dict['end_turn'])

            episodes.append(NormEpisode(
                start_turn=ep_dict['start_turn'],
                end_turn=ep_dict['end_turn'],
                description=categorization['description'],
                episode_description=categorization['episode_description'],
                participants=ep_dict['participants'],
                episode_type=categorization['episode_type'],
                is_explicit=categorization['is_explicit'],
                norm_id=None,
                turn_count=ep_dict['turn_count'],
                episode_text=episode_text,
                turn_explanations=ep_dict.get('turn_explanations', [])
            ))

    # Step 5: Link norms
    episodes, norms = link_norms(episodes)

    # Step 6: Create chunks with adjusted boundaries
    chunks = create_norm_chunks(total_turns, episodes, chunk_size)

    # Step 7: Aggregate per-chunk metrics
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

    Uses a 2-step LLM approach:
    1. Per-turn identification of norm-related activity
    2. Categorize each episode (type, explicit/implicit, norm description)

    Episodes are formed by batching consecutive norm-related turns.
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
        cache_dir: Directory for caching (default: data/processed/norms_results_v3)
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
        cache_dir = str(repo_root / 'data' / 'processed' / 'norms_results_v3')

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
