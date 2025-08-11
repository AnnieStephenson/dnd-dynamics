"""
Multi-Provider Prompt Caching for D&D Simulation

This module provides unified prompt caching across OpenAI, Anthropic, and Google models
through LiteLLM, optimizing API costs and response times for repeated content.

Key Features:
- Provider-specific caching strategies
- Shared game context caching
- Per-character context caching  
- Rolling history window caching
- Automatic cache management
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import litellm
import anthropic
import os
from .api_config import validate_api_key_for_model, get_model_provider


def generate_system_cache() -> str:
    """
    Generate the static D&D system prompt that's shared across all characters.
    This explains the game context and response format.
    
    Returns:
        Static D&D system prompt text
    """
    return """You are participating in a Dungeons & Dragons play-by-post forum game simulation.

GAME CONTEXT:
- This is a turn-based roleplaying game where players control fantasy characters
- Each player posts actions and dialogue for their character in response to game situations
- The Dungeon Master (DM) describes scenarios, environments, and NPC interactions
- Players should respond in character, matching typical play-by-post D&D forum style
- Responses should include both actions and dialogue as appropriate for the situation

RESPONSE GUIDELINES:
- Stay true to your character's personality, abilities, and background
- Consider the current situation and respond appropriately
- Match the posting style and tone of play-by-post D&D forums
- Keep responses focused and engaging (typically 1-3 paragraphs)
- Include both narrative description and character dialogue as needed"""


def generate_character_cache(character, include_player_personalities: bool = True) -> str:
    """
    Generate character-specific cached content including personality and stats.
    
    Args:
        character: CharacterAgent instance
        include_player_personalities: Whether to include player personality info
        
    Returns:
        Character-specific prompt text
    """
    player_section = f"""
PLAYER IDENTITY:
- Username: {character.player_name}
- Player Personality: {character.player_personality}
""" if include_player_personalities else ""

    player_style_text = f", while also staying true to the play style representative of {character.player_name}" if include_player_personalities else ""

    character_context = f"""{player_section}
CHARACTER IDENTITY:
- Character Name: {character.name}
- Character Race: {character.race}
- Character Class: {character.dnd_class}
- Character Gender: {character.gender}
- Character Personality: {character.personality}
- Character Sheet: {json.dumps(character.character_sheet, indent=2)}

ROLEPLAY INSTRUCTIONS:
You are {character.player_name} playing as {character.name}. Respond in character with what {character.name} does or says in the current situation{player_style_text}.

Your character's response should reflect their personality, abilities, and the current game state."""

    return character_context


def generate_history_cache(game_log: Dict, start_turn: int, end_turn: int) -> str:
    """
    Generate cached historical game context from a range of turns.
    
    Args:
        game_log: Complete game log dictionary
        start_turn: Starting turn number (inclusive)
        end_turn: Ending turn number (inclusive)
        
    Returns:
        Formatted historical context
    """
    if start_turn > end_turn:
        return ""
        
    history_entries = []
    for turn_num in sorted([int(k) for k in game_log.keys() 
                           if k.isdigit() and start_turn <= int(k) <= end_turn]):
        turn_data = game_log[str(turn_num)]
        character = turn_data.get('character', 'Unknown')
        
        # Extract text from paragraphs
        text_content = ""
        if 'paragraphs' in turn_data:
            for para_key, para_data in turn_data['paragraphs'].items():
                if isinstance(para_data, dict) and 'text' in para_data:
                    text_content += para_data['text'] + " "
        
        if text_content.strip():
            history_entries.append(f"Turn {turn_num} - {character}: {text_content.strip()}")
    
    return "\n".join(history_entries)


def format_current_situation(character_name: str) -> str:
    """
    Generate the current turn prompt (not cached - changes every turn).
    
    Args:
        character_name: Name of the character whose turn it is
        
    Returns:
        Current situation prompt
    """
    return f"CURRENT SITUATION:\nIt is now your turn. What does {character_name} do or say?"


def build_anthropic_messages(system_cache: str, character_cache: str, 
                           history_cache: str, recent_context: str, 
                           current_situation: str) -> List[Dict]:
    """
    Build messages array with Anthropic's cache_control format.
    Restructured so character info doesn't invalidate history cache.
    
    Cache hierarchy: System Prompt → History → Character Info → Recent Events → Current Situation
    
    Args:
        system_cache: Static D&D system prompt (cached)
        character_cache: Character-specific context (not cached - changes per character)
        history_cache: Historical game events (cached)
        recent_context: Recent game events (not cached)
        current_situation: Current turn prompt (not cached)
        
    Returns:
        Messages array with cache_control parameters
    """
    # System message with only the static D&D prompt (cached)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_cache,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    
    # Build user message content in optimal order for caching
    user_content = []
    
    # Add cached history if available (cached - shared across characters)
    if history_cache:
        user_content.append({
            "type": "text",
            "text": f"PREVIOUS GAME HISTORY:\n{history_cache}",
            "cache_control": {"type": "ephemeral"}
        })
    
    # Add character context (not cached - changes per character)
    user_content.append({
        "type": "text",
        "text": f"CHARACTER CONTEXT:\n{character_cache}"
    })
    
    # Add recent context (not cached)
    if recent_context:
        user_content.append({
            "type": "text",
            "text": f"RECENT GAME EVENTS:\n{recent_context}"
        })
    
    # Add current situation (not cached)
    user_content.append({
        "type": "text", 
        "text": current_situation
    })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    return messages


def build_gemini_messages(system_cache: str, character_cache: str,
                         history_cache: str, recent_context: str,
                         current_situation: str) -> List[Dict]:
    """
    Build messages array with Gemini's cache_control format (same as Anthropic).
    Uses TTL for longer cache duration.
    
    Args:
        system_cache: Static D&D system prompt
        character_cache: Character-specific context  
        history_cache: Historical game events (cached)
        recent_context: Recent game events (not cached)
        current_situation: Current turn prompt (not cached)
        
    Returns:
        Messages array with cache_control parameters
    """
    content_blocks = []
    
    # Add cached system prompt with longer TTL
    content_blocks.append({
        "type": "text",
        "text": system_cache,
        "cache_control": {
            "type": "ephemeral",
            "ttl": "3600s"  # 1 hour cache for shared context
        }
    })
    
    # Add cached character context  
    content_blocks.append({
        "type": "text",
        "text": character_cache,
        "cache_control": {
            "type": "ephemeral", 
            "ttl": "1800s"  # 30 minutes cache for character context
        }
    })
    
    messages = [
        {
            "role": "system",
            "content": content_blocks
        }
    ]
    
    # Build user message content
    user_content = []
    
    # Add cached history if available
    if history_cache:
        user_content.append({
            "type": "text",
            "text": f"PREVIOUS GAME HISTORY:\n{history_cache}",
            "cache_control": {
                "type": "ephemeral",
                "ttl": "600s"  # 10 minutes cache for history
            }
        })
    
    # Add recent context (not cached)
    if recent_context:
        user_content.append({
            "type": "text",
            "text": f"RECENT GAME EVENTS:\n{recent_context}"
        })
    
    # Add current situation (not cached)
    user_content.append({
        "type": "text",
        "text": current_situation
    })
    
    messages.append({
        "role": "user", 
        "content": user_content
    })
    
    return messages


def build_openai_messages(system_cache: str, character_cache: str,
                         history_cache: str, recent_context: str, 
                         current_situation: str) -> List[Dict]:
    """
    Build messages array optimized for OpenAI's automatic prefix caching.
    Places all static content at the beginning for optimal cache hits.
    
    Args:
        system_cache: Static D&D system prompt
        character_cache: Character-specific context
        history_cache: Historical game events (for prefix caching)  
        recent_context: Recent game events (dynamic)
        current_situation: Current turn prompt (dynamic)
        
    Returns:
        Messages array optimized for OpenAI prefix caching
    """
    # Combine all static content in system message for prefix caching
    system_content = system_cache + "\n\n" + character_cache
    
    messages = [
        {
            "role": "system",
            "content": system_content
        }
    ]
    
    # Build user message with history first (for prefix caching)
    user_parts = []
    
    if history_cache:
        user_parts.append(f"PREVIOUS GAME HISTORY:\n{history_cache}")
    
    if recent_context:
        user_parts.append(f"RECENT GAME EVENTS:\n{recent_context}")
        
    user_parts.append(current_situation)
    
    messages.append({
        "role": "user",
        "content": "\n\n".join(user_parts)
    })
    
    return messages


def build_cached_messages(provider: str, system_cache: str, character_cache: str,
                         history_cache: str, recent_context: str, 
                         current_situation: str) -> List[Dict]:
    """
    Main function to build cached messages for any provider.
    
    Args:
        provider: Provider name ("anthropic", "openai", "google")
        system_cache: Static D&D system prompt
        character_cache: Character-specific context
        history_cache: Historical game events (cached when appropriate)
        recent_context: Recent game events (dynamic)  
        current_situation: Current turn prompt (dynamic)
        
    Returns:
        Provider-optimized messages array
    """
    if provider == "anthropic":
        return build_anthropic_messages(
            system_cache, character_cache, history_cache, 
            recent_context, current_situation
        )
    elif provider == "google":
        return build_gemini_messages(
            system_cache, character_cache, history_cache,
            recent_context, current_situation  
        )
    elif provider == "openai":
        return build_openai_messages(
            system_cache, character_cache, history_cache,
            recent_context, current_situation
        )
    else:
        # Fallback to simple message format for any unrecognized provider
        return [
            {
                "role": "system", 
                "content": f"{system_cache}\n\n{character_cache}"
            },
            {
                "role": "user",
                "content": f"{history_cache}\n\n{recent_context}\n\n{current_situation}"
            }
        ]


class HistoryCacheManager:
    """
    Manages the rolling window cache of game history for the D&D simulation.
    """
    
    def __init__(self, cache_update_interval: int = 50):
        """
        Initialize history cache manager.
        
        Args:
            cache_update_interval: How many turns before updating history cache
        """
        self.cache_update_interval = cache_update_interval
        self.history_cache = ""
        self.last_history_update = 0
    

    
    def should_update_history_cache(self, current_turn: int) -> bool:
        """
        Determine if the history cache needs updating.
        
        Args:
            current_turn: Current turn number
            
        Returns:
            True if history cache should be updated
        """
        # Update every N turns
        return current_turn - self.last_history_update >= self.cache_update_interval
    
    def update_history_cache(self, current_turn: int, game_log: Dict):
        """
        Update the cached historical context.
        
        Args:
            current_turn: Current turn number
            game_log: Complete game log
        """
        if current_turn <= self.cache_update_interval:
            self.history_cache = ""
            return
            
        # Cache everything except the most recent turns
        history_end = current_turn - self.cache_update_interval
        self.history_cache = generate_history_cache(game_log, 0, history_end)
        self.last_history_update = current_turn
    
    def get_recent_context(self, current_turn: int, game_log: Dict) -> str:
        """
        Get recent game context (not cached).
        
        Args:
            current_turn: Current turn number
            game_log: Complete game log
            
        Returns:
            Recent context string
        """
        start_turn = max(0, current_turn - self.cache_update_interval)
        return generate_history_cache(game_log, start_turn, current_turn - 1)


def create_cached_completion(character, game_log: Dict, current_turn: int,
                           history_cache_manager: HistoryCacheManager, 
                           include_player_personalities: bool = True,
                           print_cache: bool = False,
                           **kwargs) -> str:
    """
    Main entry point for creating cached completions.
    
    Args:
        character: CharacterAgent instance
        game_log: Complete game log dictionary
        current_turn: Current turn number
        history_cache_manager: HistoryCacheManager instance
        include_player_personalities: Whether to include player info
        print_cache: Whether to print cache statistics after API call
        **kwargs: Additional arguments for litellm.completion
        
    Returns:
        Character's response text
    """
    # Validate API key
    validate_api_key_for_model(character.model)
    
    # Detect provider
    provider = get_model_provider(character.model)
    
    # Update history cache if needed
    if history_cache_manager.should_update_history_cache(current_turn):
        history_cache_manager.update_history_cache(current_turn, game_log)
    
    # Generate content components
    system_cache = generate_system_cache()  # Generate fresh each time
    character_cache = generate_character_cache(character, include_player_personalities)
    history_cache = history_cache_manager.history_cache
    recent_context = history_cache_manager.get_recent_context(current_turn, game_log)
    current_situation = format_current_situation(character.name)
    
    # Build provider-optimized messages
    messages = build_cached_messages(
        provider, system_cache, character_cache, 
        history_cache, recent_context, current_situation
    )
    
    # Prepare completion arguments
    completion_args = {
        "model": character.model,
        "messages": messages,
        "max_tokens": 300,
        **kwargs
    }
    
    # Add provider-specific parameters
    if provider == "openai":
        completion_args["prompt_cache_key"] = f"dnd_character_{character.name}"
    
    # Debug: Print message structure for first few turns
    if current_turn <= 3:
        print(f"🔧 [DEBUG T{current_turn}] Model: {character.model}")
        print(f"🔧 [DEBUG T{current_turn}] Provider: {provider}")
        print(f"🔧 [DEBUG T{current_turn}] Message structure:")
        for i, msg in enumerate(messages):
            print(f"   Message {i}: Role={msg['role']}")
            if isinstance(msg['content'], list):
                print(f"   Content blocks: {len(msg['content'])}")
                for j, block in enumerate(msg['content']):
                    cache_info = block.get('cache_control', 'No cache')
                    print(f"     Block {j}: {cache_info}")
            else:
                print(f"   Content: String (length={len(msg['content'])})")
    
    # Make API call - use direct Anthropic API for cache support
    if provider == "anthropic":
        response = create_anthropic_cached_completion(character, messages, **kwargs)
        
        # Print cache statistics if requested
        if print_cache:
            print_anthropic_cache_statistics(character.name, current_turn, response)
        
        return response.content[0].text.strip()
    else:
        # Use LiteLLM for other providers
        response = litellm.completion(**completion_args)
        
        # Print cache statistics if requested
        if print_cache:
            print_cache_statistics(character.name, current_turn, response, provider)
        
        return response.choices[0].message.content.strip()
    
    # OLD CODE (using LiteLLM for all providers):
    # response = litellm.completion(**completion_args)
    # if print_cache:
    #     print_cache_statistics(character.name, current_turn, response, provider)
    # return response.choices[0].message.content.strip()


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count for debugging purposes.
    Anthropic tokens are roughly 3.5-4 characters on average.
    """
    return len(text) // 4


def create_anthropic_cached_completion(character, messages, **kwargs):
    """
    Direct Anthropic API call with proper cache_control support.
    
    Args:
        character: CharacterAgent instance
        messages: Messages array with cache_control
        **kwargs: Additional arguments
        
    Returns:
        Anthropic response object
    """
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    
    # Extract system message and user messages
    system_content = None
    user_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_content = msg['content']
        else:
            user_messages.append(msg)
    
    # Debug: Print what we're sending to Anthropic
    print(f"🔧 [DEBUG] Sending to Anthropic API:")
    print(f"   System content type: {type(system_content)}")
    
    total_cached_text = ""
    if isinstance(system_content, list):
        print(f"   System blocks: {len(system_content)}")
        for i, block in enumerate(system_content):
            cache_info = block.get('cache_control', 'No cache_control')
            text_length = len(block.get('text', ''))
            estimated_tokens = estimate_token_count(block.get('text', ''))
            print(f"     Block {i}: cache_control = {cache_info}, chars = {text_length}, est_tokens = {estimated_tokens}")
            if 'cache_control' in block:
                total_cached_text += block.get('text', '')
    
    print(f"   User messages: {len(user_messages)}")
    for i, msg in enumerate(user_messages):
        if isinstance(msg.get('content'), list):
            print(f"     User msg {i} blocks: {len(msg['content'])}")
            for j, block in enumerate(msg['content']):
                cache_info = block.get('cache_control', 'No cache_control')
                text_length = len(block.get('text', ''))
                estimated_tokens = estimate_token_count(block.get('text', ''))
                print(f"       Block {j}: cache_control = {cache_info}, chars = {text_length}, est_tokens = {estimated_tokens}")
                if 'cache_control' in block:
                    total_cached_text += block.get('text', '')
    
    # Show total cached content estimation
    if total_cached_text:
        total_cached_chars = len(total_cached_text)
        total_cached_est_tokens = estimate_token_count(total_cached_text)
        print(f"   📊 Total cached content: {total_cached_chars} chars, ~{total_cached_est_tokens} tokens")
        if total_cached_est_tokens < 1024:
            print(f"   ⚠️  ISSUE: Estimated {total_cached_est_tokens} cached tokens < 1024 minimum!")
    
    # Make direct Anthropic API call with cache_control support
    response = client.messages.create(
        model=character.model,
        max_tokens=kwargs.get('max_tokens', 300),
        system=system_content,
        messages=user_messages
    )
    
    return response


def print_anthropic_cache_statistics(character_name: str, turn_number: int, response):
    """
    Print cache statistics from direct Anthropic API response.
    
    Args:
        character_name: Name of the character
        turn_number: Current turn number
        response: Direct Anthropic response object
    """
    print(f"🔧 [{character_name} T{turn_number}] Detected Provider: anthropic (direct API)")
    
    # Debug: Print all available fields in the response
    print(f"🔧 [DEBUG] Response type: {type(response)}")
    print(f"🔧 [DEBUG] Response attributes: {dir(response)}")
    
    # Anthropic direct API response structure
    usage = response.usage
    if not usage:
        print(f"🔧 [{character_name} T{turn_number}] No usage statistics available")
        return
    
    # Debug: Print all available usage fields
    print(f"🔧 [DEBUG] Usage type: {type(usage)}")
    print(f"🔧 [DEBUG] Usage attributes: {[attr for attr in dir(usage) if not attr.startswith('_')]}")
    
    # Try to get all attributes from usage
    usage_dict = {}
    for attr in dir(usage):
        if not attr.startswith('_'):
            try:
                value = getattr(usage, attr)
                if not callable(value):
                    usage_dict[attr] = value
            except:
                pass
    
    print(f"🔧 [DEBUG] All usage fields: {usage_dict}")
    
    # Basic token counts
    total_tokens = usage.input_tokens + usage.output_tokens
    prompt_tokens = usage.input_tokens
    completion_tokens = usage.output_tokens
    
    print(f"🔧 [{character_name} T{turn_number}] Total: {total_tokens}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    # Check minimum token requirement for caching
    if prompt_tokens < 1024:
        print(f"   ⚠️  WARNING: Only {prompt_tokens} input tokens - minimum 1024 required for caching on Claude Sonnet models")
    
    # Try multiple possible cache field names
    cached_tokens = getattr(usage, 'cache_read_input_tokens', 0) or getattr(usage, 'cached_tokens', 0)
    cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or getattr(usage, 'cache_creation_tokens', 0)
    
    print(f"   💾 Cache Read: {cached_tokens}, Cache Creation: {cache_creation_tokens}")
    
    # Show success message if we see cache activity
    if cached_tokens > 0:
        cache_hit_rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
        print(f"   ✅ Cache working! {cache_hit_rate:.1f}% of prompt cached")
    elif cache_creation_tokens > 0:
        print(f"   🔄 Cache created for future use ({cache_creation_tokens} tokens)")
    else:
        print(f"   ⚠️  No cache activity - troubleshooting:")
        print(f"       - Check if prompt has ≥1024 tokens (current: {prompt_tokens})")
        print(f"       - Verify cache_control blocks are identical between calls")
        print(f"       - Ensure subsequent calls are within 5 minutes")
        print(f"       - Try using exact model name instead of 'latest' variant")


def print_cache_statistics(character_name: str, turn_number: int, response, provider: str):
    """
    Print cache statistics from API response for debugging.
    
    Args:
        character_name: Name of the character
        turn_number: Current turn number
        response: LiteLLM response object
        provider: Provider name ("anthropic", "openai", "google")
    """
    # Debug: Print provider detection
    print(f"🔧 [{character_name} T{turn_number}] Detected Provider: {provider}")
    
    # Extract usage statistics
    usage = response.usage
    if not usage:
        print(f"🔧 [{character_name} T{turn_number}] No usage statistics available")
        return
    
    # Debug: Print all available usage fields
    print(f"🔧 [{character_name} T{turn_number}] Available usage fields: {dir(usage)}")
    
    # Basic token counts
    total_tokens = getattr(usage, 'total_tokens', 0)
    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
    completion_tokens = getattr(usage, 'completion_tokens', 0)
    
    print(f"🔧 [{character_name} T{turn_number}] Total: {total_tokens}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    # Provider-specific cache statistics
    if provider == "anthropic":
        # Get cache info from prompt_tokens_details
        prompt_details = getattr(usage, 'prompt_tokens_details', None)
        cached_tokens = 0
        cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
        
        if prompt_details and hasattr(prompt_details, 'cached_tokens'):
            cached_tokens = prompt_details.cached_tokens
            
        print(f"   💾 Cache Read: {cached_tokens}, Cache Creation: {cache_creation_tokens}")
        
        # Show if caching is configured but not working
        if cached_tokens == 0:
            print(f"   ⚠️  Caching appears to be configured but not working (possibly LiteLLM compatibility issue)")
            
    elif provider == "openai":
        # OpenAI cache fields (automatic prefix caching)
        cached_tokens = getattr(usage, 'prompt_tokens_cached', 0)
        cache_hit_rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
        print(f"   💾 Cached Tokens: {cached_tokens} ({cache_hit_rate:.1f}% cache hit)")
            
    elif provider == "google":
        # Google/Gemini cache fields
        cached_content_tokens = getattr(usage, 'cached_content_token_count', 0)
        candidates_token_count = getattr(usage, 'candidates_token_count', 0)
        print(f"   💾 Cached Content: {cached_content_tokens}, Candidates: {candidates_token_count}")
    
    else:
        print(f"   ℹ️  Provider '{provider}' cache stats not specifically handled")
