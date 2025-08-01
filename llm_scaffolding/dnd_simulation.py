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

from .api_config import validate_api_key_for_model
from analysis import data_loading as dl
# ===================================================================
# CAMPAIGN PARAMETER EXTRACTION
# ===================================================================

def extract_campaign_parameters(campaign_file_path: str, model: str = "claude-3-5-sonnet-20240620") -> Dict[str, Any]:
    """
    Load human campaign file and extract initialization parameters.

    Args:
        campaign_file_path: Path to individual campaign JSON file
        
    Returns:
        Dictionary containing campaign parameters:
        - num_players: Number of unique players
        - character_names: List of character names (for future use)
        - campaign_name: Name of the campaign
        - total_messages: Total number of messages in campaign
    """
    # Load and label data
    with open(campaign_file_path, 'r', encoding='utf-8') as f:
        campaign_data = json.load(f)
    campaign_name = Path(campaign_file_path).stem
    single_campaign_data = {campaign_name: campaign_data}
    df = dl.load_dnd_data(single_campaign_data)

    # Extract metadata
    character_turns = np.array(df['character'].tolist())
    character_turns = character_turns[character_turns != None]
    character_names = np.array(df['character'].unique().tolist())
    character_names = character_names[character_names != None]

    character_classes = [list(df[df['character'] == name]['class'])[0] for name in character_names]
    character_races = [list(df[df['character'] == name]['race'])[0] for name in character_names]
    character_genders = [list(df[df['character'] == name]['gender'])[0] for name in character_names]
    player_names = [list(df[df['character'] == name]['player'])[0] for name in character_names]

    num_players = len(character_names)
    total_messages = len(df)

    validate_api_key_for_model(model)

    character_personalities = generate_character_personalities(
        campaign_data=campaign_data,
        character_names=character_names,
        model=model)

    player_personalities = generate_player_personalities(
        campaign_data=campaign_data,
        player_names=player_names,
        model=model)
    
    character_sheets = generate_character_sheets(
        campaign_data=campaign_data,
        character_names=character_names,
        model=model) 

    # Extract the initial scenario from the Dungeon Master
    scenario_text = {}
    current_char = 'Dungeon Master'
    i = 1
    while current_char == 'Dungeon Master':
        scenario_text[str(i)] = campaign_data[str(i)]
        i += 1
        current_char = campaign_data[str(i)]['character']

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

def generate_character_personalities(campaign_data: Dict[str, Any],
                                     character_names: List[str],
                                     model: str) -> List[str]:
    """
    Query LLM to extract fictional character personalities and backstories.
    """

    prompt = f"""
        You are analyzing a Dungeons & Dragons play-by-post campaign.

        Your task is to generate a rich, detailed personality and backstory summary for each character, based on how they are portrayed by the human player — especially in early posts, dialogue, and actions.

        Use all available information to describe:
        - Their personality traits (e.g., brave, secretive, idealistic)
        - Backstory elements (e.g., origin, motivations, relationships)
        - Role in the group or story
        - Any quirks, values, or unique traits

        If the character is well-developed, your response may be 10+ sentences. 

        Campaign data:
        {json.dumps(campaign_data, indent=2)}

        Characters found:
        {np.array(character_names[character_names!='Dungeon Master'])}

        For each character, format your response like this on a single line:

        [Character Name]:[Detailed fictional character personality and backstory]

        For the Dungeon Master, who is not technically a character, 
        """

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    response_text = response.choices[0].message.content
    lines = [line.strip() for line in response_text.strip().split('\n') if ':' in line]
    valid_lines = [line.strip() for line in lines if any(line.strip().startswith(name + ":") for name in character_names)]
    personalities = []

    line_idx = 0
    for name in character_names:
        if name == "Dungeon Master":
            personalities.append(None)
        else:
            name_in_line, desc = valid_lines[line_idx].split(':', 1)
            personalities.append(desc.strip())
            line_idx += 1

    return personalities

def generate_player_personalities(campaign_data: Dict[str, Any],
                                     player_names: List[str],
                                     model: str) -> List[str]:
    """
    Query LLM to extract fictional character personalities and backstories.
    """

    prompt = f"""
        You are analyzing the behavior and writing of players in a Dungeons & Dragons play-by-post campaign.

        For each player, generate a detailed psychological profile, inferred from their writing style, gameplay decisions, social behavior, and tone of voice throughout the game.
        For this description, we are not interested in traits of the character being played, but in the human player who is roleplaying that character. 
        Be sure to separate the human player's personality from that of their character. 

        You may include:
        - Personality traits (e.g., introverted, playful, meticulous)
        - Writing style (e.g., descriptive, terse, humorous, lyrical)
        - Possible age range, gender identity (if suggested), or background
        - Hobbies, interests, or career hints
        - Political or ethical leanings (if evidenced)
        - Social tendencies (e.g., leadership, collaboration, conflict-avoidance)
        - Possible family or personal life details
        - Any other relevant psychological insights

        Only include details that are **reasonably supported** by the gameplay data — be thoughtful and cautious, but specific.

        Campaign data:
        {json.dumps(campaign_data, indent=2)}

        Characters found:
        {np.array(player_names)}

        For each player, format your response like this on a single line:

        [Player Name]:[Detailed player personality and profile]

        """

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    response_text = response.choices[0].message.content
    lines = [line.strip() for line in response_text.strip().split('\n') if ':' in line]
    valid_lines = [line.strip() for line in lines if any(line.strip().startswith(name + ":") for name in player_names)]
    player_personalities = []

    line_idx = 0
    for name in player_names:
        name_in_line, desc = valid_lines[line_idx].split(':', 1)
        player_personalities.append(desc.strip())
        line_idx += 1

    return player_personalities

def generate_character_sheets(campaign_data: Dict[str, Any],
                             character_names: List[str],
                             model: str) -> Dict[str, Dict]:
    """
    Query LLM to extract and infer complete D&D character sheets from campaign text.
    
    Args:
        campaign_data: Campaign message data
        character_names: List of character names
        model: LLM model to use
        
    Returns:
        Dictionary mapping character names to their character sheet dictionaries
    """
    
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

        Campaign data:
        {json.dumps(campaign_data, indent=2)}

        Characters to analyze:
        {np.array(character_names[character_names != 'Dungeon Master'])}

        IMPORTANT: Use ability scores and level from the initial game state, not from later progression during the campaign.

        For some campaigns, the character sheet may include additional parameters containing qualitative questions and answers from the DM, such as "why are you here?" or "character background". If these aspects are provided, include them in your response.

        For each character, provide a complete character sheet in this exact format:

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

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000
    )

    response_text = response.choices[0].message.content
    character_sheets = [None] # start with DM character sheet of none
    
    # Parse response into character sheets
    current_character = None
    current_sheet = {}
    
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new character
        if any(line.startswith(name + ":") for name in character_names if name != "Dungeon Master"):
            # Save previous character if exists
            if current_character and current_sheet:
                character_sheets.append(current_sheet.copy())
            
            # Start new character
            current_character = line.split(':', 1)[0].strip()
            current_sheet = {}
            continue
        
        # Parse character sheet fields
        if ':' in line and current_character:
            field_name, field_value = line.split(':', 1)
            field_name = field_name.strip()
            field_value = field_value.strip()
            
            # Convert certain fields to appropriate types
            if field_name in ['Level', 'Strength', 'Dexterity', 'Constitution', 
                             'Intelligence', 'Wisdom', 'Charisma', 'Armor Class']:
                try:
                    if field_value.lower() != 'unknown':
                        field_value = int(field_value)
                except ValueError:
                    pass  # Keep as string if conversion fails
            
            elif field_name in ['Saving Throw Proficiencies', 'Skill Proficiencies', 
                               'Languages', 'Equipment', 'Spells Known', 'Special Abilities']:
                # Convert comma-separated lists
                if field_value.lower() != 'unknown':
                    field_value = [item.strip() for item in field_value.split(',') if item.strip()]
            
            current_sheet[field_name] = field_value
    
    # Don't forget the last character
    if current_character and current_sheet:
        character_sheets.append(current_sheet.copy())
    
    return character_sheets

# ===============================================================
# CHARACTER CREATION
# ===============================================================


def create_characters(campaign_params: Dict, model: str = "claude-3-5-sonnet-20240620") -> List['CharacterAgent']:
    """
    Generate D&D characters for the simulation.
    
    Args:
        character_data: extracted character info from human campaign
        
    Returns:
        List of CharacterAgent objects
    """
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
                 character_sheet: Dict, model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize character agent.
        
        Args:
            name: Character name
            personality: Character personality description
            model: LLM model to use
        """
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

    def generate_response(self, game_log: str) -> str:
        """
        Generate character's action/dialogue for current situation.
        
        Args:
            game_log: Game log so far
            
        Returns:
            Character's response/action
        """
        prompt = f"""You are roleplaying in a Dungeons & Dragons play-by-post forum game.

        PLAYER IDENTITY:
        - Username: {self.player_name}

        CHARACTER IDENTITY:
        - Character Name: {self.name}
        - Character Personality: {self.personality}
        - Character Sheet: {self.character_sheet}

        CURRENT GAME STATE:
        {game_log}

        INSTRUCTIONS:
        You are {self.player_name} playing as {self.name}. Respond in character with what {self.name} does or says in this situation.

        Your response should:
        - Stay true to {self.name}'s personality and abilities, while also staying to the play style representative of {self.player_name}
        - Be appropriate for the current situation
        - Include both actions and/or dialogue as needed
        - Match the posting style typical of play-by-post D&D forums

        {self.name}'s response:"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Warning: Failed to generate response for {self.name}: {e}")
            return f"{self.name} pauses, considering the situation carefully."


# ===================================================================
# GAME SESSION CLASS
# ===================================================================


class GameSession:
    """
    Main game session manager that orchestrates the D&D simulation.
    """

    def __init__(self, characters: List[CharacterAgent]):
        """
        Initialize game session.
        
        Args:
            characters: List of CharacterAgent objects
        """
        self.characters = characters
        self.game_log = {}
        self.turn_counter = 0

        # Create character lookup
        self.character_lookup = {char.name: char for char in characters}

    def execute_turn(self, character_name: str):
        """
        Execute a turn for the specified character.
        
        Args:
            character_name: Name of character taking the turn
        """
        character = self.character_lookup[character_name]

        # Generate character response
        response = character.generate_response(self.game_log)

        # Log the event
        self.log_event(character, response)

        # Print the response
        print(f"\n{character_name}: {response}")

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
        print('turn counter: ' + str(self.turn_counter))
        print('\n')
        self.game_log[str(self.turn_counter)] = event
        self.turn_counter += 1

    def run_scenario(self, initial_scenario: str, turn_sequence: List[str]):
        """
        Main game loop that iterates through the turn sequence.
        
        Args:
            initial_scenario: Starting scenario description
            turn_sequence: List of character names in turn order
        """
        # Set initial scene
        self.game_log.update(initial_scenario)
        print(f"=== INITIAL GAME LOG ===")
        print(self.game_log)

        self.turn_counter = int(max(self.game_log.keys(), key=int)) + 1
        print(self.turn_counter)
        print(f"=== D&D SIMULATION STARTING ===")

        print(f"Characters: {[char.name for char in self.characters]}")
        print(f"Total turns: {len(turn_sequence)}")
        print("=" * 50)

        # Execute turns
        for character_name in turn_sequence:
            self.execute_turn(character_name)

        print("\n" + "=" * 50)
        print("=== SIMULATION COMPLETE ===")
