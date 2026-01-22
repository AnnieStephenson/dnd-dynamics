"""
D&D human dynamics and LLM gameplay package
"""

__version__ = "0.0.0"

# Global LLM parameters as a mutable config object
class _Config:
    DEFAULT_MAX_TOKENS = 3000
    DEFAULT_TEMPERATURE = 1.0
    SIMULATION_MODEL = "gemini/gemini-2.0-flash"  # For LLM game simulations
    CORRECTION_MODEL = "gemini/gemini-2.0-flash"  # For data corrections
    EXTRACTION_INITIAL_TURNS = 100
    EXTRACTION_INTRO_WINDOW = 50
    EXTRACTION_PRE_INTRO_TURNS = 10

    # Sentence embedding model for cohesion metrics
    SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Can use "all-mpnet-base-v2" for MPNet
    SENTENCE_EMBEDDING_BATCH_SIZE = 64

    # Session grouping
    MESSAGES_PER_SESSION = 5

config = _Config()
