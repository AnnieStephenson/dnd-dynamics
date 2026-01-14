"""
D&D LLM gameplay simulation module


"""

__version__ = "0.0.0"

# Global LLM parameters as a mutable config object
class _Config:
    DEFAULT_MAX_TOKENS = 3000
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_MODEL = "gemini/gemini-1.5-pro"

config = _Config()