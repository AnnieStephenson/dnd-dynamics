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
import anthropic

# Add analysis directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "analysis"))
import dnd_analysis as ana

MODEL = "claude-3-5-sonnet-20240620" #"claude-3-haiku-20240307"
# ===================================================================
# CAMPAIGN PARAMETER EXTRACTION
# ===================================================================


def extract_campaign_parameters(campaign_file_path: str) -> Dict[str, Any]:
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
    # Load campaign data
    with open(campaign_file_path, 'r', encoding='utf-8') as f:
        campaign_data = json.load(f)

    # Extract campaign name from filename
    campaign_name = Path(campaign_file_path).stem

    # Create single-campaign data structure
    single_campaign_data = {campaign_name: campaign_data}

    # Load as DataFrame to analyze
    df = ana.load_dnd_data(single_campaign_data)

    # Extract parameters
    character_turns = np.array(df['character'].tolist())
    character_names = np.array(df['character'].unique().tolist())
    character_names = character_names[character_names
                                      != None]  # Remove empty names
    character_classes = []
    character_races = []
    character_genders = []
    player_names = []
    for char_name in character_names:
        character_classes.append(
            list(df[df['character'] == char_name]['class'])[0])
        character_races.append(
            list(df[df['character'] == char_name]['race'])[0])
        character_genders.append(
            list(df[df['character'] == char_name]['gender'])[0])
        player_names.append(
            list(df[df['character'] == char_name]['player'])[0])

    num_players = len(character_names)
    total_messages = len(df)

    # Extract personality descriptions using Anthropic API
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Create prompt for personality extraction
    prompt = f"""
    Analyze this D&D play-by-post campaign data and provide a personality description for each character.
    
    Campaign data: {json.dumps(campaign_data, indent=2)}
    
    Characters found: {list(character_names)}
    
    For each character, provide a 2-3 sentence personality description based on their posting style, 
    interactions, and gameplay approach. Focus on the human player's personality, not the fictional character.
    
    Format your response as:
    Character1: [personality description]
    Character2: [personality description]
    etc.
    """

    response_char_personality = client.messages.create(model=MODEL,
                                                       max_tokens=1000,
                                                       messages=[{
                                                           "role":
                                                           "user",
                                                           "content":
                                                           prompt
                                                       }])

    # Parse the response to extract personalities
    character_personalities = []
    response_text_char_person = response_char_personality.content[0].text

    for line in response_text_char_person.split('\n'):
        if ':' in line:
            char_name, personality = line.split(':', 1)
            char_name = char_name.strip()
            personality = personality.strip()
            if char_name in character_names:
                character_personalities.append(personality)

    # Extract the initial scenario
    scenario_text = {}
    current_char = 'Dungeon Master'
    i = 1
    while current_char == 'Dungeon Master':
        text = campaign_data[str(i)]
        scenario_text[str(i)] = text
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
        'character_turns': character_turns,
        'initial_scenario': scenario_text
    }


# ===============================================================
# CHARACTER CREATION
# ===============================================================


def create_characters(campaign_params: Dict) -> List['CharacterAgent']:
    """
    Generate D&D characters for the simulation.
    
    Args:
        character_data: extracted character info from human campaign
        
    Returns:
        List of CharacterAgent objects
    """
    characters = []
    api_key = os.getenv('ANTHROPIC_API_KEY')

    num_players = campaign_params['num_players']
    # Create characters based on available templates
    for i in range(num_players):
        # Use extracted character data
        char_name = campaign_params['character_names'][i]
        char_personality = campaign_params['character_personalities'][i]
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
                                   api_key=api_key)
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
                 dnd_class: str, personality: str, api_key: str):
        """
        Initialize character agent.
        
        Args:
            name: Character name
            personality: Character personality description
            api_key: Anthropic API key
        """
        self.name = name
        self.player_name = player_name
        self.gender = gender
        self.race = race
        self.dnd_class = dnd_class
        self.combat_bool = False
        self.personality = personality
        self.api_key = api_key

        # Initialize memory
        self.memory_summary = f"I am {name}, and I am playing D&D with my fellow adventurers. My personality can be described like this: {personality} "

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(self, game_log: str) -> str:
        """
        Generate character's action/dialogue for current situation.
        
        Args:
            game_log: Game log so far
            
        Returns:
            Character's response/action
        """
        prompt = f"""You are playing Dungeons and Dragons on a play-by-post forum. You are a person playing with username {self.player_name}, 
            
            playing a character named: {self.name} with a personality described like this: {self.personality}

            The game so far looks like this:
            {game_log}

            -----------------------------
            It is now your turn to post. As {self.name}, what do you do or say in this situation? Respond in character with actions and/or dialogue. 

            Your response:"""

        try:
            response = self.client.messages.create(model=MODEL,
                                                   max_tokens=300,
                                                   messages=[{
                                                       "role": "user",
                                                       "content": prompt
                                                   }])

            return response.content[0].text.strip()

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

    def __init__(self, characters: List[CharacterAgent], api_key: str):
        """
        Initialize game session.
        
        Args:
            characters: List of CharacterAgent objects
            api_key: Anthropic API key
        """
        self.characters = characters
        self.api_key = api_key
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
