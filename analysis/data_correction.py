"""
Campaign Data Error Correction

This module provides automated and manual correction functions for common data quality 
issues in D&D campaign datasets, including:
- Removing accidental single-post characters  
- Resolving duplicate character name conflicts
- Applying manual corrections from configuration file
- LLM-based intelligent name correction

Author: Claude Code Assistant
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd

# Import LLM functionality
import sys
sys.path.append(str(Path(__file__).parent.parent))
import litellm
import time

# Local constants to avoid circular import
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TEMPERATURE = 1.0


def _retry_llm_call(func, *args, max_retries=3, initial_delay=10, **kwargs):
    """Simple retry wrapper for LLM calls to avoid circular import."""
    retry_delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(keyword in error_str for keyword in [
                '502', 'bad gateway', 'service unavailable', '503', 
                'timeout', 'connection error', 'server error', '500',
                '529', 'overloaded', 'rate limit', 'disconnected'
            ])
            
            if is_retryable and attempt < max_retries - 1:
                print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 1.5
                continue
            else:
                raise e


def load_manual_corrections(corrections_file: str) -> Dict:
    """
    Load manual corrections from JSON file.
    
    Args:
        corrections_file: Path to corrections JSON file
        
    Returns:
        Dictionary of manual corrections by campaign
    """
    if not os.path.exists(corrections_file):
        print(f"📝 Manual corrections file not found: {corrections_file}")
        return {}
    
    try:
        with open(corrections_file, 'r', encoding='utf-8') as f:
            corrections = json.load(f)
        print(f"📚 Loaded manual corrections for {len(corrections)} campaigns")
        return corrections
    except Exception as e:
        print(f"⚠️ Error loading manual corrections: {e}")
        return {}


def remove_single_post_characters(campaign_data: Dict) -> Dict:
    """
    Remove characters that only appear in a single post (likely accidental).
    
    Args:
        campaign_data: Raw campaign data dictionary
        
    Returns:
        Cleaned campaign data with single-post characters removed
    """
    corrected_data = {}
    
    # Count posts per character
    character_counts = defaultdict(int)
    for post_id, post_data in campaign_data.items():
        if isinstance(post_data, dict) and 'character' in post_data:
            character = post_data['character']
            if character and character != 'Dungeon Master':
                character_counts[character] += 1
    
    # Identify single-post characters
    single_post_chars = [char for char, count in character_counts.items() if count == 1]
    
    if single_post_chars:
        print(f"🗑️ Removing single-post characters: {single_post_chars}")
    
    # Filter out posts from single-post characters
    for post_id, post_data in campaign_data.items():
        if isinstance(post_data, dict) and 'character' in post_data:
            character = post_data['character']
            if character not in single_post_chars:
                corrected_data[post_id] = post_data
        else:
            corrected_data[post_id] = post_data
    
    return corrected_data


def detect_character_name_conflicts(campaign_data: Dict) -> List[Tuple[str, List[str]]]:
    """
    Detect cases where multiple players use the same character name.
    
    Args:
        campaign_data: Campaign data dictionary
        
    Returns:
        List of (character_name, [list_of_players]) for conflicts
    """
    character_to_players = defaultdict(set)
    
    # Map characters to their players
    for post_id, post_data in campaign_data.items():
        if isinstance(post_data, dict) and 'character' in post_data and 'player' in post_data:
            character = post_data['character']
            player = post_data['player']
            if character and player and character != 'Dungeon Master':
                character_to_players[character].add(player)
    
    # Find conflicts (character used by multiple players)
    conflicts = []
    for character, players in character_to_players.items():
        if len(players) > 1:
            conflicts.append((character, list(players)))
            print(f"⚠️ Character name conflict: '{character}' used by players {list(players)}")
    
    return conflicts


def resolve_duplicate_character_names_llm(campaign_data: Dict, 
                                        character_name: str, 
                                        players: List[str],
                                        model: str = "claude-sonnet-4-5-20250929") -> Dict[str, str]:
    """
    Use LLM to resolve duplicate character name conflicts by analyzing player posts.
    
    Args:
        campaign_data: Campaign data dictionary
        character_name: The conflicted character name
        players: List of players using this character name
        model: LLM model to use for analysis
        
    Returns:
        Dictionary mapping player -> corrected_character_name
    """
    corrections = {}
    
    for player in players:
        # Get first 10 posts from this player using the conflicted character name
        player_posts = []
        for post_id, post_data in campaign_data.items():
            if (isinstance(post_data, dict) and 
                post_data.get('player') == player and 
                post_data.get('character') == character_name):
                
                # Extract text content
                text_content = ""
                if 'paragraphs' in post_data:
                    for para_id, para_data in post_data['paragraphs'].items():
                        if isinstance(para_data, dict) and 'text' in para_data:
                            text_content += para_data['text'] + " "
                
                if text_content.strip():
                    player_posts.append({
                        'post_id': post_id,
                        'text': text_content.strip()
                    })
                    
                if len(player_posts) >= 10:  # Limit to first 10 posts
                    break
        
        if not player_posts:
            print(f"⚠️ No posts found for player {player} with character {character_name}")
            corrections[player] = character_name  # Keep original if no posts
            continue
            
        # Build prompt for LLM analysis
        posts_text = "\n".join([f"Post {p['post_id']}: {p['text']}" for p in player_posts[:5]])
        
        prompt = f"""You are analyzing a D&D campaign with a character name conflict.

Player "{player}" is using the character name "{character_name}" but there's a conflict with other players.
Based on their posts, what is this player's character's ACTUAL name?

Look for:
- Self-introductions ("I am...", "My name is...", "Call me...")
- Other characters addressing them by a different name
- Character descriptions mentioning their real name
- Consistent name usage that differs from "{character_name}"

Posts from player {player}:
{posts_text}

IMPORTANT: Return ONLY the correct character name (no explanation), or return "{character_name}" if the original name appears to be correct based on the posts."""

        try:
            response = _retry_llm_call(
                litellm.completion,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,  # Short response expected
                temperature=DEFAULT_TEMPERATURE
            )
            
            suggested_name = response.choices[0].message.content.strip()
            corrections[player] = suggested_name
            
            if suggested_name != character_name:
                print(f"🔄 Player {player}: '{character_name}' → '{suggested_name}'")
            else:
                print(f"✅ Player {player}: '{character_name}' (confirmed correct)")
                
        except Exception as e:
            print(f"⚠️ LLM correction failed for player {player}: {e}")
            corrections[player] = character_name  # Keep original on error
    
    return corrections


def apply_manual_corrections(campaign_data: Dict, 
                           campaign_name: str, 
                           manual_corrections: Dict) -> Dict:
    """
    Apply manual corrections from the corrections file.
    
    Args:
        campaign_data: Campaign data dictionary
        campaign_name: Name of the campaign
        manual_corrections: Manual corrections dictionary
        
    Returns:
        Campaign data with manual corrections applied
    """
    if campaign_name not in manual_corrections:
        return campaign_data
    
    corrections = manual_corrections[campaign_name]
    corrected_data = {}
    
    # Apply character name corrections
    char_corrections = corrections.get('character_name_corrections', {})
    if char_corrections:
        print(f"📝 Applying manual character corrections: {char_corrections}")
    
    for post_id, post_data in campaign_data.items():
        if isinstance(post_data, dict) and 'character' in post_data:
            character = post_data['character']
            if character in char_corrections:
                post_data = post_data.copy()  # Don't modify original
                post_data['character'] = char_corrections[character]
        
        corrected_data[post_id] = post_data
    
    return corrected_data


def apply_all_corrections(campaign_data: Dict, 
                         campaign_name: str,
                         manual_corrections: Dict,
                         apply_llm_corrections: bool = True,
                         model: str = "claude-sonnet-4-5-20250929") -> Dict:
    """
    Apply all automated and manual corrections to campaign data.
    
    Args:
        campaign_data: Raw campaign data dictionary
        campaign_name: Name of the campaign
        manual_corrections: Manual corrections dictionary
        apply_llm_corrections: Whether to use LLM for duplicate name resolution
        model: LLM model to use
        
    Returns:
        Fully corrected campaign data
    """
    print(f"🔧 Applying corrections to campaign: {campaign_name}")
    
    # Step 1: Apply manual corrections first
    corrected_data = apply_manual_corrections(campaign_data, campaign_name, manual_corrections)
    
    # Step 2: Remove single-post characters
    corrected_data = remove_single_post_characters(corrected_data)
    
    # Step 3: Resolve duplicate character names with LLM
    if apply_llm_corrections:
        conflicts = detect_character_name_conflicts(corrected_data)
        
        for character_name, players in conflicts:
            corrections = resolve_duplicate_character_names_llm(
                corrected_data, character_name, players, model
            )
            
            # Apply the LLM corrections
            for post_id, post_data in corrected_data.items():
                if (isinstance(post_data, dict) and 
                    post_data.get('character') == character_name and 
                    post_data.get('player') in corrections):
                    
                    player = post_data['player']
                    new_name = corrections[player]
                    if new_name != character_name:
                        post_data = post_data.copy()
                        post_data['character'] = new_name
                        corrected_data[post_id] = post_data
    
    print(f"✅ Corrections completed for campaign: {campaign_name}")
    return corrected_data