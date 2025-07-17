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
import anthropic

# Add analysis directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "analysis"))


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

    # Use existing analysis functions to extract parameters
    from dnd_analysis import load_dnd_data

    # Create single-campaign data structure
    single_campaign_data = {campaign_name: campaign_data}

    # Load as DataFrame to analyze
    df = load_dnd_data(single_campaign_data)

    # Extract parameters
    num_players = df['player'].nunique()
    character_names = df['character'].unique().tolist()
    total_messages = len(df)

    return {
        'num_players': num_players,
        'character_names': character_names,
        'campaign_name': campaign_name,
        'total_messages': total_messages
    }, df


# ===================================================================
# CHARACTER CREATION
# ===================================================================

def create_characters(num_players: int, character_data: Optional[Dict] = None) -> List['CharacterAgent']:
    """
    Generate D&D characters for the simulation.
    
    Args:
        num_players: Number of characters to create
        character_data: Optional extracted character info from human campaign
        
    Returns:
        List of CharacterAgent objects
    """
    # Default character templates
    default_characters = [
        {
            'name': 'Aria',
            'character_class': 'Fighter',
            'personality': 'Brave and decisive warrior who leads by example and protects her allies.'
        },
        {
            'name': 'Thorin',
            'character_class': 'Rogue',
            'personality': 'Cunning and stealthy scout who prefers shadows and clever solutions.'
        },
        {
            'name': 'Gandalf',
            'character_class': 'Wizard',
            'personality': 'Wise and scholarly spellcaster who seeks knowledge and magical solutions.'
        },
        {
            'name': 'Lyra',
            'character_class': 'Cleric',
            'personality': 'Compassionate healer who provides spiritual guidance and divine magic.'
        },
        {
            'name': 'Zara',
            'character_class': 'Ranger',
            'personality': 'Nature-loving archer who tracks enemies and protects the wilderness.'
        },
        {
            'name': 'Finn',
            'character_class': 'Bard',
            'personality': 'Charismatic storyteller who inspires allies and solves problems with wit.'
        }
    ]

    characters = []
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Create characters based on available templates
    for i in range(num_players):
        if character_data and i < len(character_data.get('character_names', [])):
            # Use extracted character data (future extension)
            char_name = character_data['character_names'][i]
            char_class = 'Unknown'  # Would extract from campaign data
            personality = f"A {char_class} adventurer with unique traits."
        else:
            # Use default character template
            template = default_characters[i % len(default_characters)]
            char_name = template['name']
            char_class = template['character_class']
            personality = template['personality']

        character = CharacterAgent(
            name=char_name,
            character_class=char_class,
            personality=personality,
            api_key=api_key
        )
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

    def __init__(self, name: str, character_class: str, personality: str, api_key: str):
        """
        Initialize character agent.
        
        Args:
            name: Character name
            character_class: D&D class (Fighter, Rogue, etc.)
            personality: Character personality description
            api_key: Anthropic API key
        """
        self.name = name
        self.character_class = character_class
        self.personality = personality
        self.api_key = api_key

        # Initialize memory
        self.memory_summary = f"I am {name}, a {character_class}. {personality} I am playing D&D with my fellow adventurers."

        self.client = anthropic.Anthropic(api_key=api_key)

    def update_memory(self, recent_verbatim_text: str, older_events_summary: str = ""):
        """
        Update character memory with recent events and older summary.
        
        Args:
            recent_verbatim_text: Exact recent character responses and events
            older_events_summary: Summary of older events (for future use)
        """
        # Use Claude to update memory
        prompt = f"""You are {self.name}, a {self.character_class}. Here is your current memory:

            {self.memory_summary}

            Recent events have occurred:
            {recent_verbatim_text}

            Please update your memory summary to include these recent events. Keep it concise but capture important details about what happened, who was involved, and any decisions made. Maintain your character's perspective and personality.

            Updated memory summary:"""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            self.memory_summary = response.content[0].text.strip()

        except Exception as e:
            print(f"Warning: Failed to update memory for {self.name}: {e}")
            # Fallback: simple append
            self.memory_summary += f"\n\nRecent events: {recent_verbatim_text}"

    def generate_response(self, current_situation: str) -> str:
        """
        Generate character's action/dialogue for current situation.
        
        Args:
            current_situation: Description of current game state
            
        Returns:
            Character's response/action
        """
        prompt = f"""You are {self.name}, a {self.character_class}. {self.personality}

            Your current memory and knowledge:
            {self.memory_summary}

            Current situation:
            {current_situation}

            As {self.name}, what do you do or say in this situation? Respond in character with actions and/or dialogue. Keep it concise but engaging. Include both what you do and what you say if appropriate.

            Your response:"""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

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
        self.game_log = []
        self.current_scene = ""
        self.turn_counter = 0

        # Create character lookup
        self.character_lookup = {char.name: char for char in characters}

    def get_recent_events_for_character(self, character_name: str) -> str:
        """
        Get exact character response texts since this character's last turn.
        
        Args:
            character_name: Name of character to get events for
            
        Returns:
            String of recent events since character's last turn
        """
        # Find this character's last turn
        last_turn_index = -1
        for i, event in enumerate(self.game_log):
            if event['character'] == character_name:
                last_turn_index = i

        # Get events since last turn
        recent_events = []
        for i in range(last_turn_index + 1, len(self.game_log)):
            event = self.game_log[i]
            recent_events.append(f"{event['character']}: {event['action']}")

        return "\n".join(recent_events) if recent_events else "No recent events since your last turn."

    def get_older_events_summary_for_character(self, character_name: str) -> str:
        """
        Get summarized character responses from before their last turn.
        
        Args:
            character_name: Name of character
            
        Returns:
            Summary of older events (structured for future use)
        """
        # For now, return simple summary
        # Future extension: Use Claude to summarize older events
        return f"Earlier events in the adventure involving {character_name} and companions."

    def execute_turn(self, character_name: str):
        """
        Execute a turn for the specified character.
        
        Args:
            character_name: Name of character taking the turn
        """
        if character_name not in self.character_lookup:
            print(f"Warning: Character {character_name} not found")
            return

        character = self.character_lookup[character_name]

        # Get recent events for memory update
        recent_events = self.get_recent_events_for_character(character_name)
        older_events = self.get_older_events_summary_for_character(character_name)

        # Update character memory
        if recent_events and recent_events != "No recent events since your last turn.":
            character.update_memory(recent_events, older_events)

        # Generate character response
        response = character.generate_response(self.current_scene)

        # Log the event
        self.log_event(character_name, response)

        # Print the response
        print(f"\n{character_name}: {response}")

    def log_event(self, character_name: str, action_text: str):
        """
        Record turn with timestamp and turn number.
        
        Args:
            character_name: Name of character
            action_text: What the character did/said
        """
        self.turn_counter += 1

        event = {
            'turn_number': self.turn_counter,
            'character': character_name,
            'action': action_text,
            'timestamp': datetime.now().isoformat()
        }

        self.game_log.append(event)

    def run_scenario(self, initial_scenario: str, turn_sequence: List[str]):
        """
        Main game loop that iterates through the turn sequence.
        
        Args:
            initial_scenario: Starting scenario description
            turn_sequence: List of character names in turn order
        """
        # Set initial scene
        self.current_scene = initial_scenario

        print(f"=== D&D SIMULATION STARTING ===")
        print(f"Scenario: {initial_scenario}")
        print(f"Characters: {[char.name for char in self.characters]}")
        print(f"Total turns: {len(turn_sequence)}")
        print("=" * 50)

        # Execute turns
        for character_name in turn_sequence:
            self.execute_turn(character_name)

        print("\n" + "=" * 50)
        print("=== SIMULATION COMPLETE ===")
