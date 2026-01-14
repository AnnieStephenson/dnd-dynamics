"""
D&D LLM gameplay simulation module


"""

__version__ = "0.0.0"

# Global LLM parameters as a mutable config object
class _Config:
    DEFAULT_MAX_TOKENS = 3000
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_MODEL = "gemini/gemini-1.5-pro"
    EXTRACTION_INITIAL_TURNS = 100  # First N turns to always include for extraction
    EXTRACTION_INTRO_WINDOW = 50  # Turns after new character introduction
    EXTRACTION_PRE_INTRO_TURNS = 10  # Fixed lookback before character introduction

config = _Config()