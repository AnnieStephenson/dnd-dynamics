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
import time
from .api_config import validate_api_key_for_model, get_model_provider
from . import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_MODEL
from analysis import data_loading as dl
from analysis import basic_metrics as basic


def retry_llm_call(func, *args, max_retries=3, initial_delay=10, **kwargs):
    """
    Retry wrapper for LLM API calls with exponential backoff.
    
    Args:
        func: Function to retry (e.g., litellm.completion)
        *args, **kwargs: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
        
    Returns:
        Function result
    """
    retry_delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if it's a retryable error
            error_str = str(e).lower()
            is_retryable = any(keyword in error_str for keyword in [
                '502', 'bad gateway', 'service unavailable', '503', 
                'timeout', 'connection error', 'server error', '500',
                '529', 'overloaded', 'overloaded_error', 'rate limit',
                'disconnected', 'geminiexception'
            ])
            
            if is_retryable and attempt < max_retries - 1:
                print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
                continue
            else:
                # Not retryable or final attempt - re-raise the error
                raise e


def generate_system_cache(campaign_name: str, scratchpad: bool = False) -> str:
    """
    Generate the D&D system prompt with campaign-specific post length statistics.
    This explains the game context and response format.
    
    Args:
        campaign_name: Name of the human campaign to load statistics from
        scratchpad: Whether to include scratchpad reasoning instructions
    
    Returns:
        D&D system prompt text with campaign-specific response length guidelines
    """
    # Load campaign data and analyze post length statistics
    df = dl.load_campaigns(source=campaign_name)
    basic_metrics = basic.analyze_basic_metrics({campaign_name: df})

    mean = basic_metrics[campaign_name]['post_lengths_overall']['overall'][
        'mean_words']
    median = basic_metrics[campaign_name]['post_lengths_overall']['overall'][
        'median_words']
    standard_dev = basic_metrics[campaign_name]['post_lengths_overall'][
        'overall']['std_words']

    # Build the system prompt
    system_prompt = f"""You are participating in a Dungeons & Dragons play-by-post forum game simulation.

GAME CONTEXT:
- This is a turn-based roleplaying game where players control fantasy characters
- Each player posts actions and dialogue for their character in response to game situations
- The Dungeon Master (DM) describes scenarios, environments, and NPC interactions
- Players should respond in character, matching typical play-by-post D&D forum style
- Responses should include both actions and dialogue as appropriate for the situation

CRITICAL TURN RESTRICTIONS:  
- You are generating EXACTLY ONE turn for your assigned role (either a player character OR the Dungeon Master)
- Do NOT generate responses for other characters or roles

RESPONSE GUIDELINES:
- Stay true to your character's personality, abilities, and background
- Consider the current situation and respond appropriately
- Match the posting style and tone of play-by-post D&D forums
- Include both narrative description and character dialogue as needed

RESPONSE LENGTH GUIDELINES:
Your response length should follow the distribution of post lengths in the campaign, 
but should be an appropriate length based on the narrative context. In this campaign, 
the median post length is {median:.1f} words, the mean is {mean:.1f} words, and the standard deviation is {standard_dev:.1f} words. 
Most campaigns are have a post length distribution that is right skewed with a mode smaller than the mean and median, and have a somewhat lognormal tail, 
meaning that longer posts are possible, but are relatively rare. The typical (mode) post length is therefore likely less
than {median:.1f} words. Sometimes, very short responses of just a handful of words are fine. Rarely, long responses of over 400 words might be appropriate."""

    # Add scratchpad instructions if enabled
    if scratchpad:
        scratchpad_instructions = """

REASONING PROCESS:
Before generating your character's response, you should think through your reasoning process. Consider:
- What is the current situation and what just happened?
- How would you feel about this situation given your personality?
- What actions or dialogue would be most fitting?
- Importantly, what would be an appropriate response length considering both the typical post lengths for this campaign, and considering the context?

FORMAT YOUR RESPONSE AS FOLLOWS:
First, work through your reasoning in a "thinking" section, then provide your final character response.

Reasoning: [Your analysis of the situation, character motivations, and response planning]

Final response: [Your actual character's actions and dialogue - this should be what gets posted to the forum]

IMPORTANT: 
- You MUST include "Final response:" (exactly this text) before your character's actual response
- Only the "Final response:" portion will be visible to other players and added to the game history."""

        system_prompt += scratchpad_instructions

    return system_prompt


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


def format_turns_as_text(game_log: Dict, start_turn: int, end_turn: int) -> str:
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


def generate_turn_summary(game_log: Dict, start_turn: int, end_turn: int,
                          model: str = None) -> str:
    """
    Generate a narrative summary of turns [start_turn, end_turn].
    Preserves: character decisions, plot developments, relationships, quests.

    Args:
        game_log: Complete game log dictionary
        start_turn: Starting turn number (inclusive)
        end_turn: Ending turn number (inclusive)
        model: LLM model to use for summarization

    Returns:
        Concise narrative summary of the turns
    """
    if model is None:
        model = DEFAULT_MODEL

    raw_history = format_turns_as_text(game_log, start_turn, end_turn)

    if not raw_history.strip():
        return ""

    prompt = f"""Summarize this D&D game session (turns {start_turn}-{end_turn}).
Preserve: key character decisions, plot developments, relationship changes,
quest progress, combat outcomes, important dialogue.
Keep it concise but complete enough for continuity.

GAME HISTORY:
{raw_history}

SUMMARY:"""

    response = retry_llm_call(litellm.completion,
                              model=model,
                              messages=[{"role": "user", "content": prompt}],
                              max_tokens=DEFAULT_MAX_TOKENS)
    return response.choices[0].message.content.strip()


def format_current_situation(character_name: str) -> str:
    """
    Generate the current turn prompt (not cached - changes every turn).
    
    Args:
        character_name: Name of the character whose turn it is
        
    Returns:
        Current situation prompt
    """
    return f"CURRENT SITUATION:\nIt is now {character_name}'s turn. Generate {character_name}'s single response only."


def build_anthropic_messages(system_cache: str, character_cache: str,
                           history_cache: str, recent_context: str,
                           current_situation: str) -> List[Dict]:
    """
    Build messages array with Anthropic's cache_control format with multiple cache breakpoints.
    
    Cache hierarchy with breakpoints:
    1. System Prompt (cached) - Static D&D system prompt
    2. History (cached) - Historical game events, shared across characters  
    3. Character Context (cached) - Character-specific info, cached per character
    4. Recent Context (not cached) - Recent game events, changes every turn
    5. Current Situation (not cached) - Current turn prompt, changes every turn
    
    Args:
        system_cache: Static D&D system prompt (pre-cached)
        character_cache: Character-specific context (pre-cached)
        history_cache: Historical game events (cached)
        recent_context: Recent game events (not cached)
        current_situation: Current turn prompt (not cached)
        
    Returns:
        Messages array that uses pre-cached content
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

    # Build user message content with multiple cache breakpoints
    user_content = []

    # Add cached history if available (cached - shared across characters)
    if history_cache:
        user_content.append({
            "type": "text",
            "text": f"PREVIOUS GAME HISTORY:\n{history_cache}",
            "cache_control": {"type": "ephemeral"}
        })

    # Add character context (will use pre-cached content)
    user_content.append({
        "type": "text",
        "text": f"CHARACTER CONTEXT:\n{character_cache}"
    })

    # Add recent context (not cached - changes every turn)
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
    Restructured so character info doesn't break prefix caching.
    
    Cache-friendly order: System Prompt → History → Character Info → Recent Events → Current Situation
    
    Args:
        system_cache: Static D&D system prompt (cached)
        character_cache: Character-specific context (not cached - changes per character)
        history_cache: Historical game events (for prefix caching)  
        recent_context: Recent game events (dynamic)
        current_situation: Current turn prompt (dynamic)
        
    Returns:
        Messages array optimized for OpenAI prefix caching
    """
    # Only static system prompt in system message (for prefix caching)
    messages = [
        {
            "role": "system",
            "content": system_cache
        }
    ]

    # Build user message with cacheable content first, then dynamic content
    user_parts = []

    # Add history first (most cacheable across characters)
    if history_cache:
        user_parts.append(f"PREVIOUS GAME HISTORY:\n{history_cache}")

    # Add recent context (semi-cacheable)
    if recent_context:
        user_parts.append(f"RECENT GAME EVENTS:\n{recent_context}")

    # Add character context last (changes per character, breaks prefix cache)
    user_parts.append(f"CHARACTER CONTEXT:\n{character_cache}")

    # Add current situation at the very end
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
        system_cache: Static D&D system prompt (pre-cached)
        character_cache: Character-specific context (pre-cached)
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
        # TODO: Re-enable when on paid Gemini account
        # return build_gemini_messages(
        #     system_cache, character_cache, history_cache,
        #     recent_context, current_situation
        # )

        # Temporary: Disable caching for Gemini due to free tier limitations
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
    Manages game history context for LLM prompts.

    Supports three modes:
    - Full context: verbatim_window=None passes all turns
    - Limited window: summary_chunk_size=0 passes only last N turns
    - Summarization: older turns are summarized, recent turns stay verbatim
    """

    def __init__(self,
                 summary_chunk_size: int = 50,
                 verbatim_window: int = 50,
                 summary_model: str = None):
        """
        Initialize history cache manager with summarization support.

        Args:
            summary_chunk_size: Number of turns per summary chunk (default 50).
                Set to 0 to disable summarization.
            verbatim_window: Minimum verbatim turns to keep (default 50).
                Set to None to pass all turns as context (no limit).
            summary_model: LLM model to use for summarization
        """
        self.summary_chunk_size = summary_chunk_size
        self.verbatim_window = verbatim_window
        self.summary_model = summary_model or DEFAULT_MODEL

        self.history_cache = ""           # Concatenated summaries (empty if no summarization)
        self.summaries = {}               # {chunk_start: summary_text}
        self.last_summarized_turn = -1    # Track highest summarized turn

    def _needs_new_summary(self, current_turn: int) -> bool:
        """
        Check if we should generate a new summary.

        Generate summary when BOTH conditions met:
        1. We have a complete chunk to summarize
        2. At least verbatim_window turns remain after summarizing

        Returns False (no summarization) if:
        - verbatim_window is None (all turns mode)
        - summary_chunk_size == 0 (no summarization mode)
        """
        # Summarization disabled if verbatim_window is None or chunk_size is 0
        if self.verbatim_window is None or self.summary_chunk_size == 0:
            return False

        turns_summarized = len(self.summaries) * self.summary_chunk_size
        next_chunk_end = turns_summarized + self.summary_chunk_size - 1

        # Can't summarize turns that don't exist yet
        if next_chunk_end >= current_turn:
            return False

        # Would we have enough verbatim turns remaining?
        verbatim_remaining = current_turn - (next_chunk_end + 1)
        return verbatim_remaining >= self.verbatim_window

    def _generate_next_summary(self, current_turn: int, game_log: Dict):
        """Generate the next needed summary chunk."""
        chunk_index = len(self.summaries)
        chunk_start = chunk_index * self.summary_chunk_size
        chunk_end = chunk_start + self.summary_chunk_size - 1

        print(f"\n{'='*60}")
        print(f"Generating summary for turns {chunk_start}-{chunk_end}...")
        summary = generate_turn_summary(
            game_log, chunk_start, chunk_end, self.summary_model)

        print(f"\n[Summary of turns {chunk_start}-{chunk_end}]")
        print(summary)
        print(f"{'='*60}\n")

        self.summaries[chunk_start] = summary
        self.last_summarized_turn = chunk_end

        self._rebuild_history_cache()

    def _rebuild_history_cache(self):
        """Rebuild the cached history from all summaries."""
        if not self.summaries:
            self.history_cache = ""
            return

        parts = ["=== PREVIOUS SESSION SUMMARIES ==="]
        for chunk_start in sorted(self.summaries.keys()):
            chunk_end = chunk_start + self.summary_chunk_size - 1
            parts.append(f"\n[Turns {chunk_start}-{chunk_end}]")
            parts.append(self.summaries[chunk_start])

        self.history_cache = "\n".join(parts)

    def should_generate_summary(self, current_turn: int) -> bool:
        """Public method to check if summary generation is needed."""
        return self._needs_new_summary(current_turn)

    def generate_summary(self, current_turn: int, game_log: Dict):
        """Public method to generate next summary. Call in a loop until no more needed."""
        if self._needs_new_summary(current_turn):
            self._generate_next_summary(current_turn, game_log)

    def get_recent_context(self, current_turn: int, game_log: Dict) -> str:
        """
        Get verbatim recent turns (not summarized). Does NOT trigger summarization.

        Args:
            current_turn: Current turn number
            game_log: Complete game log

        Returns:
            Recent context string

        Behavior:
        - If summaries exist: returns turns from last_summarized_turn + 1
        - If verbatim_window is None: returns all turns (no limit)
        - If verbatim_window is set (no summaries): returns only last N turns
        """
        if self.summaries:
            # Have summaries: start after last summarized turn
            verbatim_start = self.last_summarized_turn + 1
        elif self.verbatim_window is None:
            # No summaries, no limit: all turns
            verbatim_start = 0
        else:
            # No summaries, limited window: only last N turns
            verbatim_start = max(0, current_turn - self.verbatim_window)

        return format_turns_as_text(game_log, verbatim_start, current_turn - 1)


def pre_cache_static_content(characters: List, system_cache: str,
                           include_player_personalities: bool = True):
    """
    Pre-cache system prompt and all character contexts at simulation start.
    This creates caches for all static content that won't change during the game.
    
    Args:
        characters: List of CharacterAgent objects
        system_cache: System prompt text
        include_player_personalities: Whether to include player personality info
    """
    # Cache system prompt and first character context with a minimal completion
    if characters:
        first_character = characters[0]
        provider = get_model_provider(first_character.model)

        if provider == "anthropic":
            # Create minimal character cache for the first character
            character_cache = generate_character_cache(first_character, include_player_personalities)

            # Build message with both system and character caches
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
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"CHARACTER CONTEXT:\n{character_cache}",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": "Please respond with just 'Ready' to confirm you understand the context."
                        }
                    ]
                }
            ]

            # Make a minimal completion to create the caches
            try:
                response = retry_llm_call(
                    litellm.completion,
                    model=first_character.model,
                    messages=messages,
                    max_tokens=10,
                    temperature=DEFAULT_TEMPERATURE
                )
                print(f"✅ Pre-cached system prompt and character context for {first_character.name}")
            except Exception as e:
                print(f"⚠️  Pre-caching failed: {e}")

        # Pre-cache remaining character contexts
        for character in characters[1:]:
            if get_model_provider(character.model) == "anthropic":
                character_cache = generate_character_cache(character, include_player_personalities)

                messages = [
                    {
                        "role": "system",
                        "content": system_cache  # Will use existing cache
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"CHARACTER CONTEXT:\n{character_cache}",
                                "cache_control": {"type": "ephemeral"}
                            },
                            {
                                "type": "text",
                                "text": "Please respond with just 'Ready' to confirm you understand the context."
                            }
                        ]
                    }
                ]

                try:
                    response = retry_llm_call(
                        litellm.completion,
                        model=character.model,
                        messages=messages,
                        max_tokens=10,
                        temperature=DEFAULT_TEMPERATURE
                    )
                    print(f"✅ Pre-cached character context for {character.name}")
                except Exception as e:
                    print(f"⚠️  Pre-caching failed for {character.name}: {e}")


def generate_character_response(character,
                             game_log: Dict,
                             current_turn: int,
                             history_cache_manager: HistoryCacheManager,
                             system_cache: str,
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
        system_cache: Pre-generated system cache (passed from GameSession)
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

    # Generate summaries if threshold crossed
    while history_cache_manager.should_generate_summary(current_turn):
        history_cache_manager.generate_summary(current_turn, game_log)

    # Generate content components
    character_cache = generate_character_cache(character,
                                               include_player_personalities)
    history_cache = history_cache_manager.history_cache
    recent_context = history_cache_manager.get_recent_context(
        current_turn, game_log)
    current_situation = format_current_situation(character.name)

    # Build provider-optimized messages
    messages = build_cached_messages(provider, system_cache, character_cache,
                                     history_cache, recent_context,
                                     current_situation)

    # Prepare completion arguments
    completion_args = {
        "model": character.model,
        "messages": messages,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        **kwargs
    }

    # Add provider-specific parameters
    if provider == "openai":
        # OpenAI automatic prefix caching - may not work with LiteLLM
        # Try without explicit cache key first to see if automatic caching works
        pass  # Let OpenAI handle automatic prefix caching

    # Use LiteLLM for all providers with retry logic
    response = retry_llm_call(litellm.completion, **completion_args)

    # Print cache statistics if requested
    if print_cache:
        print_cache_statistics(character.name, current_turn, response,
                               provider)

    return response.choices[0].message.content.strip()


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count for debugging purposes.
    Anthropic tokens are roughly 3.5-4 characters on average.
    """
    return len(text) // 4


def print_cache_statistics(character_name: str, turn_number: int, response, provider: str):
    """
    Print cache statistics from API response for debugging.
    
    Args:
        character_name: Name of the character
        turn_number: Current turn number
        response: LiteLLM response object
        provider: Provider name ("anthropic", "openai", "google")
    """

    # Extract usage statistics
    usage = response.usage
    if not usage:
        print(f"🔧 [{character_name} T{turn_number}] No usage statistics available")
        return

    # Basic token counts
    total_tokens = getattr(usage, 'total_tokens', 0)
    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
    completion_tokens = getattr(usage, 'completion_tokens', 0)

    print(f"[{character_name} T{turn_number}] Total tokens: {total_tokens}, Prompt toekns: {prompt_tokens}, Completion tokens: {completion_tokens}")

    # Provider-specific cache statistics
    if provider == "anthropic":
        # Get cache info from prompt_tokens_details
        prompt_details = getattr(usage, 'prompt_tokens_details', None)
        cached_tokens = 0
        cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)

        if prompt_details and hasattr(prompt_details, 'cached_tokens'):
            cached_tokens = prompt_details.cached_tokens

        print(f"   💾 Cache Read: {cached_tokens}, Cache Creation: {cache_creation_tokens}")

    elif provider == "openai":
        # OpenAI cache fields via LiteLLM (follows OpenAI format)
        prompt_tokens_details = getattr(usage, 'prompt_tokens_details', None)
        cached_tokens = 0

        if prompt_tokens_details and hasattr(prompt_tokens_details, 'cached_tokens'):
            cached_tokens = prompt_tokens_details.cached_tokens

        cache_hit_rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
        print(f"   💾 Cached Tokens: {cached_tokens} ({cache_hit_rate:.1f}% cache hit)")

        # Debug: Show if we're above minimum token threshold
        if prompt_tokens < 1024:
            print(f"   ⚠️  WARNING: Only {prompt_tokens} prompt tokens - OpenAI requires ≥1024 for caching")

    elif provider == "google":
        # Google/Gemini cache fields
        cached_content_tokens = getattr(usage, 'cached_content_token_count', 0)
        candidates_token_count = getattr(usage, 'candidates_token_count', 0)
        print(f"   💾 Cached Content: {cached_content_tokens}, Candidates: {candidates_token_count}")

    else:
        print(f"   ℹ️  Provider '{provider}' cache stats not specifically handled")
