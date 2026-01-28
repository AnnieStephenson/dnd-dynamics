"""
LLM-based D&D Simulation System

This module provides a complete system for simulating D&D gameplay using LLMs.
It includes campaign parameter extraction, character creation, turn sampling,
and game session management with memory-aware character agents.

Author: Claude Code Assistant
"""

import json
import random
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path
import numpy as np
import litellm
import textwrap
import re

from dnd_dynamics.api_config import validate_api_key_for_model, retry_llm_call
from dnd_dynamics import config
from . import prompt_caching as pc
from .prompt_caching import format_turns_as_text
from dnd_dynamics.analysis import data_loading as dl


def find_character_first_appearances(raw_campaign_json: Dict, character_names: List[str]) -> Dict[str, int]:
    """
    Find the first turn where each character appears.

    Args:
        raw_campaign_json: Full campaign data dictionary
        character_names: List of character names to find

    Returns:
        Dict mapping character name to first turn number
    """
    first_appearances = {}
    message_keys = sorted([int(k) for k in raw_campaign_json.keys() if k.isdigit()])

    for key_num in message_keys:
        key_str = str(key_num)
        character = raw_campaign_json[key_str].get('character')
        if character in character_names and character not in first_appearances:
            first_appearances[character] = key_num
            if len(first_appearances) == len(character_names):
                break

    return first_appearances


def build_extraction_excerpts(
    raw_campaign_json: Dict,
    character_names: List[str],
    initial_turns: int = None,
    intro_window: int = None,
    pre_intro_turns: int = None
) -> str:
    """
    Build labeled excerpts for character/player extraction.

    Includes first N turns plus windows around late-joining characters.

    Args:
        raw_campaign_json: Full campaign data dictionary
        character_names: List of character names
        initial_turns: First N turns to always include (default: config.EXTRACTION_INITIAL_TURNS)
        intro_window: Turns after new character intro (default: config.EXTRACTION_INTRO_WINDOW)
        pre_intro_turns: Fixed lookback before intro (default: config.EXTRACTION_PRE_INTRO_TURNS)

    Returns:
        Formatted string with labeled excerpts
    """
    initial_turns = initial_turns or config.EXTRACTION_INITIAL_TURNS
    intro_window = intro_window or config.EXTRACTION_INTRO_WINDOW
    pre_intro_turns = pre_intro_turns or config.EXTRACTION_PRE_INTRO_TURNS

    first_appearances = find_character_first_appearances(raw_campaign_json, character_names)
    message_keys = sorted([int(k) for k in raw_campaign_json.keys() if k.isdigit()])
    max_turn = max(message_keys) if message_keys else 0

    # Build list of (start, end, label) ranges
    ranges = [(0, min(initial_turns - 1, max_turn), "Campaign Start")]

    for char_name, first_turn in first_appearances.items():
        if char_name == "Dungeon Master":
            continue
        if first_turn >= initial_turns:
            start = max(0, first_turn - pre_intro_turns)
            end = min(first_turn + intro_window, max_turn)
            ranges.append((start, end, f"Introduction of {char_name}"))

    # Sort by start turn
    ranges.sort(key=lambda x: x[0])

    # Merge overlapping/adjacent ranges
    merged = []
    for start, end, label in ranges:
        if merged and start <= merged[-1][1] + 1:
            prev_start, prev_end, prev_label = merged[-1]
            new_label = prev_label if label in prev_label else f"{prev_label}, {label}"
            merged[-1] = (prev_start, max(prev_end, end), new_label)
        else:
            merged.append((start, end, label))

    # Build excerpts with labels as clean text
    excerpt_parts = []
    for start, end, label in merged:
        excerpt_text = format_turns_as_text(raw_campaign_json, start, end)
        if excerpt_text:
            excerpt_parts.append(f"=== TURNS {start}-{end} ({label}) ===\n{excerpt_text}")

    return "\n\n".join(excerpt_parts)


# ===================================================================
# CAMPAIGN PARAMETER EXTRACTION
# ===================================================================

def extract_campaign_parameters(campaign_file_path: str,
                                model: str = None) -> Dict[str, Any]:
    """
    Extract campaign parameters and character data from a human campaign file.

    Uses LLM to extract character personalities, player personalities, and
    character sheets from campaign excerpts.

    Args:
        campaign_file_path: Path to individual campaign JSON file
        model: LLM model to use for extraction (default: config.SIMULATION_MODEL)

    Returns:
        Dictionary containing:
        - num_players: Number of player characters
        - total_messages: Total messages in campaign
        - campaign_name: Name of the campaign
        - character_names: Array of character names
        - player_names: List of player usernames
        - character_classes/races/genders: Lists of character attributes
        - character_personalities: LLM-generated personality descriptions
        - player_personalities: LLM-generated player profiles
        - character_sheets: LLM-generated character sheet data
        - character_turns: Array of which character acted each turn
        - initial_scenario: Opening DM posts
    """
    model = model or config.SIMULATION_MODEL
    # Load and label data with corrections applied
    campaign_name = Path(campaign_file_path).stem
    campaigns, json_data = dl.load_campaigns([campaign_name], return_json=True)
    df = campaigns[campaign_name]
    raw_campaign_json = json_data[campaign_name]  # Use corrected JSON data

    # Extract metadata
    character_turns = np.array(df['character'].tolist())
    character_turns = character_turns[character_turns != None]
    character_names = np.array(df['character'].unique().tolist())
    print(character_names)
    character_names = character_names[character_names != None]

    character_classes = [list(df[df['character'] == name]['class'])[0] for name in character_names]
    character_races = [list(df[df['character'] == name]['race'])[0] for name in character_names]
    character_genders = [list(df[df['character'] == name]['gender'])[0] for name in character_names]
    player_names = [list(df[df['character'] == name]['player'])[0] for name in character_names]

    num_players = len(character_names)
    total_messages = len(df)

    validate_api_key_for_model(model)

    # Build excerpts once for all extraction functions
    excerpt_text = build_extraction_excerpts(raw_campaign_json, character_names)
    print('Campaign excerpts for extraction:')
    print(excerpt_text)

    character_personalities = generate_character_personalities(
        excerpt_text=excerpt_text,
        character_names=character_names,
        model=model)

    player_personalities = generate_player_personalities(
        excerpt_text=excerpt_text,
        player_names=player_names,
        model=model)

    character_sheets = generate_character_sheets(
        excerpt_text=excerpt_text,
        character_names=character_names,
        model=model)

    # Extract the initial scenario from the Dungeon Master
    scenario_text = {}
    current_char = 'Dungeon Master'

    # Get all message keys sorted numerically
    message_keys = sorted([int(k) for k in raw_campaign_json.keys() if k.isdigit()])

    for key_num in message_keys:
        key_str = str(key_num)
        current_char = raw_campaign_json[key_str]['character']

        if current_char == 'Dungeon Master':
            scenario_text[key_str] = raw_campaign_json[key_str]
            scenario_text[key_str]['date'] = datetime.now().isoformat()
        else:
            # Stop when we reach the first non-DM character
            break


    return {
        'num_players': num_players,
        'total_messages': total_messages,
        'campaign_name': campaign_name,
        'character_names': character_names,
        'player_names': player_names,
        'character_classes': character_classes,
        'character_races': character_races,
        'character_genders': character_genders,
        'character_personalities': character_personalities,
        'player_personalities': player_personalities,
        'character_sheets': character_sheets,
        'character_turns': character_turns,
        'initial_scenario': scenario_text
    }


def parse_name_descriptions(response_text, names):
    """
    Parse name-description pairs from LLM response formats.
    Assumes all formats use colon separators between names and descriptions.
    
    Handles formats like:
    - name1: description1
    - **name1**: description1
    - Line-separated entries
    - Handles commas within descriptions properly
    
    Args:
        response_text (str): The raw LLM response text
        names (list): List of names to look for (e.g., character names, player names)
    
    Returns:
        dict: Dictionary mapping names to their descriptions
    """
    response_text = response_text.strip()
    personalities_dict = {}

    # Strategy 1: Line-by-line parsing (handles line-separated entries)
    lines = [
        line.strip() for line in response_text.split('\n') if line.strip()
    ]

    for line in lines:
        if ':' not in line:
            continue

        # Split on first colon only
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        potential_name = parts[0].strip()
        description = parts[1].strip()

        # Clean up potential name (remove markdown formatting, bullets, etc.)
        clean_name = re.sub(r'^[\d\.\)\-\*\s]+', '', potential_name)
        clean_name = re.sub(r'\*+', '', clean_name).strip()

        # Remove parenthetical additions: "player (character)" -> "player"
        if ' (' in clean_name and ')' in clean_name:
            clean_name = clean_name.split(' (')[0].strip()

        # Check if this matches any of our names (case insensitive)
        for name in names:
            if name.lower() == clean_name.lower():
                personalities_dict[name] = description
                break

    # Strategy 2: Handle comma-separated entries (only if comma appears BEFORE a colon)
    # This catches cases like "name1: desc1, name2: desc2" but not "name1: desc with, comma in it"
    if len(personalities_dict) < len(
        [name for name in names if name != "Dungeon Master"]):
        # Look for patterns where commas separate name:description pairs
        # Use regex to find comma-separated entries that each contain name:description
        comma_pattern = r'([^,]+:[^,]*(?:,[^:]*)*?)(?=,\s*\w+\s*:|$)'
        matches = re.findall(comma_pattern, response_text)

        for match in matches:
            segment = match.strip()
            if ':' not in segment:
                continue

            parts = segment.split(':', 1)
            if len(parts) != 2:
                continue

            potential_name = parts[0].strip()
            description = parts[1].strip()

            # Clean up potential name
            clean_name = re.sub(r'^[\d\.\)\-\*\s]+', '', potential_name)
            clean_name = re.sub(r'\*+', '', clean_name).strip()

            # Remove parenthetical additions: "player (character)" -> "player"
            if ' (' in clean_name and ')' in clean_name:
                clean_name = clean_name.split(' (')[0].strip()

            # Check if this matches any of our names
            for name in names:
                if name.lower() == clean_name.lower():
                    if name not in personalities_dict:  # Don't overwrite
                        personalities_dict[name] = description
                    break

    return personalities_dict


def generate_character_personalities(excerpt_text: str,
                                     character_names: List[str],
                                     model: str) -> List[str]:
    """
    Query LLM to extract fictional character personalities and backstories.

    Args:
        excerpt_text: Pre-built campaign excerpts
        character_names: List of character names
        model: LLM model to use
    """
    prompt = f"""
        You are analyzing a Dungeons & Dragons play-by-post campaign.

        Your task is to generate a rich, detailed personality and backstory summary for each character, based on how they are portrayed by the human player — especially in early posts, dialogue, and actions.

        Use all available information to describe:
        - Their personality traits (e.g., brave, secretive, idealistic)
        - Backstory elements (e.g., origin, motivations, relationships)
        - Role in the group or story
        - Any quirks, values, or unique traits

        If the character is well-developed, your response may be 200 words or more.

        Campaign excerpts:
        {excerpt_text}

        Characters found:
        {np.array(character_names[character_names!='Dungeon Master'])}

        For each character, format your response like this on a single line, adding a blank line between each character:

        [Character Name]:[Detailed fictional character personality and backstory]

        """
    response = retry_llm_call(litellm.completion,
                             model=model,
                             messages=[{
                                 "role": "user",
                                 "content": prompt
                             }],
                             max_tokens=config.DEFAULT_MAX_TOKENS,
                             temperature=config.DEFAULT_TEMPERATURE)

    response_text = response.choices[0].message.content
    print(response_text)
    # Parse the personalities using the general parser
    personalities_dict = parse_name_descriptions(response_text,
                                                 character_names)

    # Convert to list in the same order as character_names
    personalities = []
    for name in character_names:
        if name == "Dungeon Master":
            personalities.append(None)
        else:
            personalities.append(personalities_dict.get(name, None))

    return personalities


def generate_player_personalities(excerpt_text: str,
                                  player_names: List[str],
                                  model: str) -> List[str]:
    """
    Query LLM to extract player personalities and profiles.

    Args:
        excerpt_text: Pre-built campaign excerpts
        player_names: List of player usernames
        model: LLM model to use
    """
    prompt = f"""
        You are analyzing the behavior and writing of players in a Dungeons & Dragons play-by-post campaign.

        For each player, generate a detailed psychological profile, inferred from their writing style, gameplay decisions, social behavior, and tone of voice throughout the game.
        Pay particular attention to details revelead in their out-of-character posts, as these should reaveal more about the human player. For this description, we are not interested
        in traits of the character being played, but in the human player who is roleplaying that character.
        Be sure to separate the human player's personality from that of their character.

        You may include:
        - Possible age range, gender identity, or background
        - Possible family or personal life details
        - Personality traits (e.g., introverted, playful, meticulous)
        - Writing style (e.g., descriptive, terse, humorous, lyrical)
        - Hobbies, interests, or career hints
        - Political or ethical leanings (if evidenced)
        - Social tendencies (e.g., leadership, collaboration, conflict-avoidance)
        - Any other relevant psychological insights

        Only include details that are **reasonably supported** by the gameplay data — be thoughtful and cautious, but specific.

        Campaign excerpts:
        {excerpt_text}

        Players found:
        {np.array(player_names)}

        For each player, format your response like this on a single line:

        [Player Name]:[Detailed player personality and profile]

        """

    response = retry_llm_call(litellm.completion,
                              model=model,
                              messages=[{
                                  "role": "user",
                                  "content": prompt
                              }],
                              max_tokens=config.DEFAULT_MAX_TOKENS,
                              temperature=config.DEFAULT_TEMPERATURE)

    response_text = response.choices[0].message.content

    # Parse the personalities using the general parser
    personalities_dict = parse_name_descriptions(response_text, player_names)

    # Convert to list in the same order as character_names
    player_personalities = []
    for name in player_names:
        player_personalities.append(personalities_dict.get(name, None))

    return player_personalities


def generate_character_sheets(excerpt_text: str,
                              character_names: List[str],
                              model: str = None) -> Dict[str, Dict]:
    """
    Query LLM to extract and infer complete D&D character sheets from campaign text.

    Args:
        excerpt_text: Pre-built campaign excerpts
        character_names: List of character names
        model: LLM model to use

    Returns:
        List of character sheet dictionaries
    """
    model = model or config.SIMULATION_MODEL

    prompt = f"""
        You are analyzing a Dungeons & Dragons play-by-post campaign to extract character sheet information.

        Your task is to create complete D&D character sheets for each character based on:
        1. Explicit stats mentioned in early posts (levels, abilities, etc.)
        2. Equipment and spells mentioned throughout the campaign
        3. Actions taken that reveal class abilities or proficiencies
        4. Combat descriptions that show hit points, armor class, etc.
        5. Any other character sheet details that can be reasonably inferred

        For stats not explicitly mentioned, make reasonable inferences based on:
        - Character class and typical stat distributions
        - Actions they take successfully/unsuccessfully
        - Spells they cast or abilities they use
        - Equipment they wield effectively

        Campaign excerpts:
        {excerpt_text}

        Characters to analyze:
        {np.array(character_names[character_names != 'Dungeon Master'])}
        

        IMPORTANT: Use ability scores and level from the initial game state, not from later progression during the campaign.

        For some campaigns, the character sheet may include additional parameters containing qualitative questions and answers from the DM, such as "why are you here?" or "character background". If these aspects are provided, include them in your response.

        For each character, provide a complete character sheet in this exact format with no additional formatting characters such as ** or --. Simply skip a line at the end of each character sheet.:

        [Character Name]:
        Level: [number]
        Class: [class name]
        Race: [race name]
        Background: [background if mentioned or inferred]
        Alignment: [alignment if mentioned or inferred]
        Strength: [score]
        Dexterity: [score]  
        Constitution: [score]
        Intelligence: [score]
        Wisdom: [score]
        Charisma: [score]
        Hit Points: [current/max if known]
        Armor Class: [number]
        Proficiency Bonus: [+number]
        Saving Throw Proficiencies: [list]
        Skill Proficiencies: [list]
        Languages: [list]
        Equipment: [weapons, armor, tools, etc.]
        Spells Known: [list of spells if applicable]
        Special Abilities: [class features, racial traits, etc.]
        Notes: [any other relevant character details]

        If a field cannot be determined even with reasonable inference, write "Unknown".
        Base ability scores on typical arrays (15,14,13,12,10,8) adjusted for race and class.
        """

    response = retry_llm_call(litellm.completion,
                              model=model,
                              messages=[{
                                  "role": "user",
                                  "content": prompt
                              }],
                              max_tokens=config.DEFAULT_MAX_TOKENS,
                              temperature=config.DEFAULT_TEMPERATURE)

    response_text = response.choices[0].message.content
    print(response_text)
    character_sheets = [None]  # start with DM character sheet of none

    # Parse response into character sheets
    current_character = None
    current_sheet = {}

    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this line starts a new character (handle parenthetical additions)
        line_matches_character = False
        for name in character_names:
            if name == "Dungeon Master":
                continue
            # Check for exact match: "Name:" or "[Name]:"
            if line.startswith(name + ":") or line.startswith(f"[{name}]:"):
                line_matches_character = True
                break
            # Check for match with parentheses: "Name (......):" or "[Name (......):]"
            if (line.startswith(name + " (") and "):" in line) or \
               (line.startswith(f"[{name} (") and "):" in line):
                line_matches_character = True
                break

        if line_matches_character:  # Save previous character if exists
            if current_character and current_sheet:
                character_sheets.append(current_sheet.copy())

            # Start new character - extract base name without parentheses
            name_part = line.split(':', 1)[0].strip()
            # Remove brackets if present: [Name] -> Name
            if name_part.startswith('[') and name_part.endswith(']'):
                name_part = name_part[1:-1]
            # Remove parenthetical part: "Name (extra)" -> "Name"
            if ' (' in name_part:
                name_part = name_part.split(' (')[0]
            current_character = name_part.strip()
            current_sheet = {}
            continue

        # Parse character sheet fields
        if ':' in line and current_character:
            field_name, field_value = line.split(':', 1)
            field_name = field_name.strip()
            field_value = field_value.strip()

            # Convert certain fields to appropriate types
            if field_name in [
                    'Level', 'Strength', 'Dexterity', 'Constitution',
                    'Intelligence', 'Wisdom', 'Charisma', 'Armor Class'
            ]:
                try:
                    if field_value.lower() != 'unknown':
                        field_value = int(field_value)
                except ValueError:
                    pass  # Keep as string if conversion fails

            elif field_name in [
                    'Saving Throw Proficiencies', 'Skill Proficiencies',
                    'Languages', 'Equipment', 'Spells Known',
                    'Special Abilities'
            ]:
                # Convert comma-separated lists
                if field_value.lower() != 'unknown':
                    field_value = [
                        item.strip() for item in field_value.split(',')
                        if item.strip()
                    ]

            current_sheet[field_name] = field_value

    # Don't forget the last character
    if current_character and current_sheet:
        character_sheets.append(current_sheet.copy())

    return character_sheets

# ===============================================================
# CHARACTER CREATION
# ===============================================================


def create_characters(campaign_params: Dict, model: str = None) -> List['CharacterAgent']:
    """
    Generate D&D characters for the simulation.

    Args:
        character_data: extracted character info from human campaign

    Returns:
        List of CharacterAgent objects
    """
    model = model or config.SIMULATION_MODEL
    characters = []

    num_players = campaign_params['num_players']
    # Create characters based on available templates
    for i in range(num_players):
        # Use extracted character data
        char_name = campaign_params['character_names'][i]
        char_personality = campaign_params['character_personalities'][i]
        player_personality = campaign_params['player_personalities'][i]
        character_sheet = campaign_params['character_sheets'][i]
        player_name = campaign_params['player_names'][i]
        gender = campaign_params['character_genders'][i]
        race = campaign_params['character_races'][i]
        dnd_class = campaign_params['character_classes'][i]
        character = CharacterAgent(name=char_name,
                                   player_name=player_name,
                                   gender=gender,
                                   race=race,
                                   dnd_class=dnd_class,
                                   personality=char_personality,
                                   player_personality=player_personality,
                                   character_sheet = character_sheet,
                                   model=model)
        characters.append(character)

    return characters


# ===================================================================
# TURN SAMPLING
# ===================================================================

def sample_turn_sequence(character_names: List[str], total_turns: int,
                        method: str = 'uniform') -> List[str]:
    """
    Generate sequence of character names for each turn.
    
    Args:
        character_names: List of character names
        total_turns: Total number of turns to generate
        method: Sampling method ('uniform' for now, extensible for future)
        
    Returns:
        List of character names in turn order
    """
    if method == 'uniform':
        # Uniform random sampling across all characters
        return random.choices(character_names, k=total_turns)
    else:
        # Future extension point for other methods
        # (activity-based, weighted, etc.)
        raise ValueError(f"Unknown sampling method: {method}")


# ===================================================================
# CHARACTER AGENT CLASS
# ===================================================================


class CharacterAgent:
    """
    Individual character agent with LLM-based decision making and memory.
    """

    def __init__(self, name: str, player_name: str, gender: str, race: str,
                 dnd_class: str, personality: str, player_personality: str,
                 character_sheet: Dict, model: str = None):
        """
        Initialize character agent.

        Args:
            name: Character name
            personality: Character personality description
            model: LLM model to use
        """
        model = model or config.SIMULATION_MODEL
        self.name = name
        self.player_name = player_name
        self.gender = gender
        self.race = race
        self.dnd_class = dnd_class
        self.combat_bool = False
        self.personality = personality
        self.player_personality = player_personality,
        self.character_sheet = character_sheet,
        self.model = model

        # Validate API key is available for this model
        validate_api_key_for_model(model)

        # Initialize memory
        self.memory_summary = f"I am {name}, and I am playing D&D with my fellow adventurers. My personality can be described like this: {personality} "

# ===================================================================
# GAME SESSION CLASS
# ===================================================================


class GameSession:
    """
    Main game session manager that orchestrates the D&D simulation.
    """

    def __init__(self,
                 characters: List[CharacterAgent],
                 campaign_name: str,
                 scratchpad: bool = False,
                 summary_chunk_size: int = 50,
                 verbatim_window: int = 50,
                 summary_model: str = None):
        """
        Initialize game session.

        Args:
            characters: List of CharacterAgent objects
            campaign_name: Name of the human campaign to load statistics from
            scratchpad: Whether to enable scratchpad reasoning for character responses
            summary_chunk_size: Number of turns per summary chunk (default: 50).
                Set to 0 to disable summarization.
            verbatim_window: Minimum verbatim turns to keep (default: 50).
                Set to None to pass all turns as context (no limit).
            summary_model: LLM model to use for summarization (default: config.SIMULATION_MODEL)
        """
        self.characters = characters
        self.campaign_name = campaign_name
        self.game_log = {}
        self.turn_counter = 0
        self.scratchpad = scratchpad
        self.scratchpad_log = {}  # {turn_number: reasoning_text}
        self.include_player_personalities = None  # Set during run_scenario
        self.model = config.SIMULATION_MODEL
        self.temperature = config.DEFAULT_TEMPERATURE
        self.max_tokens = config.DEFAULT_MAX_TOKENS
        self.history_cache_manager = pc.HistoryCacheManager(
            summary_chunk_size=summary_chunk_size,
            verbatim_window=verbatim_window,
            summary_model=summary_model)

        # Generate system cache once at initialization with campaign stats
        self.system_cache = pc.generate_system_cache(campaign_name, scratchpad=scratchpad)

        # Pre-cache system prompt and all character contexts
        print("Pre-caching static content...")
        pc.pre_cache_static_content(characters, self.system_cache)

        # Create character lookup
        self.character_lookup = {char.name: char for char in characters}

    def _parse_scratchpad_response(self, raw_response: str) -> str:
        """
        Parse scratchpad response to extract only the final response part.
        
        Args:
            raw_response: Full response including reasoning and final response
            
        Returns:
            Only the final response portion
            
        Raises:
            ValueError: If no "Final response:" marker found in scratchpad mode
        """
        if not self.scratchpad:
            return raw_response

        # Look for "Final response:" marker with flexible formatting (case insensitive)
        lines = raw_response.split('\n')
        final_response_lines = []
        found_final_response = False

        for line in lines:
            # Clean line for pattern matching (remove markdown formatting)
            clean_line = line.lower().strip()
            # Remove markdown formatting: **text** -> text, *text* -> text
            clean_line = clean_line.replace('**', '').replace('*', '')

            if 'final response:' in clean_line:
                found_final_response = True
                # Find the colon position in the original (uncleaned) line
                original_lower = line.lower()
                colon_positions = []

                # Look for colon after "final response" with optional markdown
                for i, char in enumerate(original_lower):
                    if char == ':':
                        # Check if this colon comes after "final response" pattern
                        text_before_colon = original_lower[:i].replace('**', '').replace('*', '').strip()
                        if text_before_colon.endswith('final response'):
                            colon_positions.append(i)

                if colon_positions:
                    colon_pos = colon_positions[-1]  # Use the last colon found
                    remainder = line[colon_pos + 1:].strip()
                    if remainder:
                        final_response_lines.append(remainder)
                continue

            if found_final_response:
                final_response_lines.append(line)

        if final_response_lines:
            return '\n'.join(final_response_lines).strip()
        else:
            # Error instead of fallback - stop execution
            raise ValueError(f"❌ SCRATCHPAD PARSING ERROR: No 'Final response:' marker found in character response.\n"
                           f"Expected formats: 'Final response:', '**Final response**:', '*Final response*:'\n"
                           f"Raw response preview: {raw_response[:200]}...")

    def execute_turn(self,
                     character_name: str,
                     include_player_personalities=True,
                     print_cache=False,
                     max_format_retries=2):
        """
        Execute a turn for the specified character.
        
        Args:
            character_name: Name of character taking the turn
            include_player_personalities: Whether to include player personality info
            print_cache: Whether to print cache statistics after API call
            max_format_retries: Maximum retries for scratchpad formatting failures
        """
        character = self.character_lookup[character_name]

        for format_attempt in range(max_format_retries + 1):
            try:
                # Generate character response and manage prompt caching
                raw_response = pc.generate_character_response(
                    character=character,
                    game_log=self.game_log,
                    current_turn=self.turn_counter,
                    history_cache_manager=self.history_cache_manager,
                    system_cache=self.system_cache,
                    include_player_personalities=include_player_personalities,
                    print_cache=print_cache)

                # Parse scratchpad response if enabled
                final_response = self._parse_scratchpad_response(raw_response)

                # Success! Break out of retry loop
                break

            except ValueError as e:
                # Check if this is a scratchpad parsing error
                if "SCRATCHPAD PARSING ERROR" in str(e) and format_attempt < max_format_retries:
                    print(f"⚠️  Scratchpad formatting error on attempt {format_attempt + 1}/{max_format_retries + 1}")
                    print(f"🔄 Retrying turn for {character_name}...")
                    continue
                else:
                    # Either not a scratchpad error, or we've exhausted retries
                    raise e

        # Store scratchpad reasoning if enabled
        if self.scratchpad and raw_response != final_response:
            self.scratchpad_log[str(self.turn_counter)] = raw_response

        # Print the response(s)
        if self.scratchpad and raw_response != final_response:
            # Show both reasoning and final response
            print(f"\n=== {character_name} REASONING ===")
            print(textwrap.fill(raw_response, width=170))
            print(f"\n=== {character_name} FINAL RESPONSE ===")
            print(textwrap.fill(final_response, width=170))
        else:
            # Regular mode - just show the response
            print(textwrap.fill(f"\n{character_name}: {final_response}", width=170))

        # Log the event (only the final response)
        self.log_event(character, final_response)

    def log_event(self, character: CharacterAgent, action_text: str):
        """
        Record turn with timestamp and turn number.
        
        Args:
            character_name: Name of character
            action_text: What the character did/said
        """
        event = {
            'date': datetime.now().isoformat(),
            'player': character.player_name,
            'character': character.name,
            'in_combat': character.combat_bool,
            'paragraphs': {
                '0': {
                    'text': action_text,
                    'actions': [],
                    'label': 'in-character'
                }
            },
            'actions': []
        }
        if character.name != 'Dungeon Master':
            event['race'] = character.race
            event['gender'] = character.gender
            event['class'] = character.dnd_class
        print('Turn counter: ' + str(self.turn_counter))
        self.game_log[str(self.turn_counter)] = event
        self.turn_counter += 1

    def run_scenario(self,
                     initial_scenario: str,
                     turn_sequence: List[str],
                     include_player_personalities=True,
                     print_cache=False):
        """
        Main game loop that iterates through the turn sequence.

        Args:
            initial_scenario: Starting scenario description
            turn_sequence: List of character names in turn order
            include_player_personalities: Whether to include player personality info
            print_cache: Whether to print cache statistics after each API call
        """
        # Track settings for save()
        self.include_player_personalities = include_player_personalities

        # Set initial scene
        self.game_log.update(initial_scenario)

        self.turn_counter = int(max(self.game_log.keys(), key=int)) + 1

        # Print the system prompt
        print(f"=== SYSTEM PROMPT ===")
        print(textwrap.fill(self.system_cache, width=170))

        # Print the initial DM message in readable format
        initial_context = self.history_cache_manager.get_recent_context(
            self.turn_counter, self.game_log)
        print(f"\n=== INITIAL SCENARIO ===")
        print(textwrap.fill(initial_context, width=170))
        print(f"\n=== D&D SIMULATION STARTING ===")

        print(f"Characters: {[char.name for char in self.characters]}")
        print(f"Total turns: {len(turn_sequence)}")
        print("=" * 50)

        # Execute turns
        for character_name in turn_sequence:
            print("-" * 50)
            self.execute_turn(
                character_name,
                include_player_personalities=include_player_personalities,
                print_cache=print_cache)

        print("\n" + "=" * 50)
        print("=== SIMULATION COMPLETE ===")

    def save(self, output_dir: str = None):
        """
        Save game log, summaries, and scratchpads to files.

        Args:
            output_dir: Override default output directory (default: {project_root}/data/llm-games)

        Creates subfolder structure:
        - llm-games/game-logs/{base}.json
        - llm-games/summaries/{base}.txt
        - llm-games/scratchpads/{base}.txt
        - llm-games/metadata_index.json (updated)
        """
        # Determine project root and output directory
        project_root = Path(__file__).resolve().parent.parent.parent
        if output_dir:
            base_dir = Path(output_dir)
        else:
            base_dir = project_root / 'data' / 'llm-games'

        # Create subfolders if they don't exist
        game_logs_dir = base_dir / 'game-logs'
        summaries_dir = base_dir / 'summaries'
        scratchpads_dir = base_dir / 'scratchpads'
        game_logs_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir.mkdir(parents=True, exist_ok=True)
        scratchpads_dir.mkdir(parents=True, exist_ok=True)

        # Build filename base
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')

        # Sanitize model name (remove provider prefix like "gemini/")
        model_str = self.model
        if '/' in model_str:
            model_str = model_str.split('/')[-1]

        player_string = 'players' if self.include_player_personalities else 'no_players'
        scratch_string = 'scratch' if self.scratchpad else 'no_scratch'

        filename_base = f"{self.campaign_name}_{model_str}_{player_string}_{scratch_string}_{timestamp}"

        # Build metadata
        metadata = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "campaign_name": self.campaign_name,
            "include_player_personalities": self.include_player_personalities,
            "scratchpad": self.scratchpad,
            "summarization_enabled": self.history_cache_manager is not None,
            "summary_chunk_size": self.history_cache_manager.summary_chunk_size if self.history_cache_manager else None,
            "extraction_initial_turns": config.EXTRACTION_INITIAL_TURNS,
            "extraction_intro_window": config.EXTRACTION_INTRO_WINDOW,
            "extraction_pre_intro_turns": config.EXTRACTION_PRE_INTRO_TURNS,
            "timestamp": now.isoformat(),
            "year": now.year,
            "month": now.month,
            "day": now.day
        }

        # Add metadata to game_log and save
        self.game_log["_metadata"] = metadata
        game_log_path = game_logs_dir / f"{filename_base}.json"
        with open(game_log_path, 'w') as f:
            json.dump(self.game_log, f, indent=2)
        print(f"💾 Game log saved to: {game_log_path}")

        # Save summaries if available
        if self.history_cache_manager and self.history_cache_manager.summaries:
            summary_path = summaries_dir / f"{filename_base}.txt"
            summary_parts = []
            for chunk_start, summary_text in sorted(self.history_cache_manager.summaries.items()):
                chunk_end = chunk_start + self.history_cache_manager.summary_chunk_size - 1
                summary_parts.append(f"=== TURNS {chunk_start}-{chunk_end} ===\n{summary_text}")
            with open(summary_path, 'w') as f:
                f.write("\n\n".join(summary_parts))
            print(f"📝 Summaries saved to: {summary_path}")

        # Save scratchpads if available
        if self.scratchpad_log:
            scratchpad_path = scratchpads_dir / f"{filename_base}.txt"
            scratchpad_parts = []
            for turn_num, reasoning in sorted(self.scratchpad_log.items(), key=lambda x: int(x[0])):
                scratchpad_parts.append(f"=== TURN {turn_num} ===\n{reasoning}")
            with open(scratchpad_path, 'w') as f:
                f.write("\n\n".join(scratchpad_parts))
            print(f"🧠 Scratchpads saved to: {scratchpad_path}")

        # Update metadata index
        index_path = base_dir / 'metadata_index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                metadata_index = json.load(f)
        else:
            metadata_index = {}

        metadata_index[filename_base] = metadata
        with open(index_path, 'w') as f:
            json.dump(metadata_index, f, indent=2)
        print(f"📋 Metadata index updated: {index_path}")
