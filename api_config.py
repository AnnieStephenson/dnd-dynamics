"""
API Configuration and Key Management for LLM Models

Handles automatic loading of API keys from api_key.txt and sets up
environment variables for different LLM providers (Anthropic, OpenAI, Google, etc.)
"""

import os
from pathlib import Path
from typing import Optional

def setup_api_keys(api_key_file: Optional[str] = None):
    """
    Automatically load API keys from api_key.txt and set environment variables.
    Supports multiple providers in one file.
    
    Args:
        api_key_file: Optional path to API key file. If None, searches for api_key.txt
    """
    if api_key_file:
        key_file_path = Path(api_key_file)
    else:
        # Try current directory first
        key_file_path = Path(__file__).parent / "api_key.txt"
        
        if not key_file_path.exists():
            # Try parent directory (for notebooks)
            key_file_path = Path(__file__).parent.parent / "api_key.txt"
    
    if not key_file_path.exists():
        print("⚠️  api_key.txt not found. Please create it with your API keys.")
        return
    
    with open(key_file_path, 'r') as f:
        content = f.read().strip()
    
    # Parse different API keys from file
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.startswith('ANTHROPIC_API_KEY='):
            os.environ['ANTHROPIC_API_KEY'] = line.split('=', 1)[1]
            print("✅ Anthropic API key loaded")
        elif line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1]  
            print("✅ OpenAI API key loaded")
        elif line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.split('=', 1)[1]
            print("✅ Gemini API key loaded")

def get_model_provider(model: str) -> str:
    """
    Automatically determine which API key is needed based on model name.
    
    Args:
        model: Model name (e.g., "claude-3-5-sonnet-20240620", "gpt-4", "gemini-pro")
        
    Returns:
        Provider name: "anthropic", "openai", or "google"
    """
    model_lower = model.lower()
    
    if 'claude' in model_lower or 'anthropic' in model_lower:
        return 'anthropic'
    elif 'gpt' in model_lower or 'openai' in model_lower:
        return 'openai'
    elif 'gemini' in model_lower or 'google' in model_lower:
        return 'google'
    else:
        # Default to anthropic for backwards compatibility
        return 'anthropic'

def validate_api_key_for_model(model: str):
    """
    Check that the required API key is available for the specified model.
    
    Args:
        model: Model name to validate
        
    Raises:
        ValueError: If required API key is not available
    """
    provider = get_model_provider(model)
    
    key_mapping = {
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY', 
        'google': 'GEMINI_API_KEY'
    }
    
    required_key = key_mapping[provider]
    if not os.environ.get(required_key):
        raise ValueError(
            f"Model '{model}' requires {required_key} in api_key.txt\n"
            f"Add a line: {required_key}=your-key-here"
        )

# Automatically load API keys when this module is imported
setup_api_keys()