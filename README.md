# D&D Dynamics 🎲

Tools for analyzing and simulating Dungeons & Dragons gameplay using statistical methods and LLM agents.

## 🎯 What This Project Does

**Analysis Tools:**
- Analyze D&D gameplay logs to extract insights about player behavior, creativity, and narrative patterns
- Compare linguistic cohesion and semantic creativity across campaigns
- Statistical analysis across hundreds of campaigns with intelligent caching

**LLM Simulation:**
- Generate realistic D&D gameplay using LLM character agents
- Extract character personalities and backstories from human campaigns  
- Simulate multi-player D&D sessions with prompt caching optimization

## 🚀 Quick Start

### Analysis Example
```python
from dnd_dynamics.analysis import data_loading as dl
from dnd_dynamics.analysis import metrics

# Load campaigns and run all metrics
campaigns = dl.load_campaigns('human', max_campaigns=10)
results = metrics.analyze_all(campaigns)

# Access individual results
basic_stats = results['basic']
semantic_analysis = results['semantic']
jaccard_cohesion = results['jaccard']
```

### Simulation Example
```python
from dnd_dynamics.llm_scaffolding import dnd_simulation as sim
import dnd_dynamics

# Configure LLM settings
dnd_dynamics.config.SIMULATION_MODEL = "gpt-4o"
dnd_dynamics.config.DEFAULT_TEMPERATURE = 1.0

# Extract parameters from human campaign
params = sim.extract_campaign_parameters('path/to/campaign.json')

# Create LLM characters and run simulation
characters = sim.create_characters(params)
session = sim.GameSession(characters, 'campaign_name',
                          scratchpad=True,
                          summary_chunk_size=50,
                          verbatim_window=50)

# Run the simulation
session.run_scenario(params['initial_scenario'], params['character_turns'])

# Save results (game log, summaries, scratchpads)
session.save()
```

### Loading LLM Games
```python
# Load all LLM games
campaigns = dl.load_campaigns('llm')

# Filter by metadata
campaigns = dl.load_campaigns('llm', filter_by={'model': 'gpt-4o'})
```

## 📁 Project Structure

```
dnd-dynamics/
├── dnd_dynamics/          # Main Python package
│   ├── __init__.py        # Package config (SIMULATION_MODEL, CORRECTION_MODEL, etc.)
│   ├── api_config.py      # API key management for LLM providers
│   ├── analysis/          # Campaign analysis tools
│   │   ├── data_loading.py    # Load and process campaign data
│   │   ├── data_correction.py # LLM-based data correction
│   │   ├── plot_utils.py      # Visualization helpers
│   │   └── metrics/           # Campaign metrics
│   │       ├── basic.py       # Time intervals, post lengths, engagement
│   │       ├── semantic.py    # SBERT embeddings, session novelty, DSI
│   │       └── jaccard.py     # Lexical cohesion via Jaccard similarity
│   └── llm_scaffolding/   # LLM simulation system
│       ├── dnd_simulation.py  # Character agents & game sessions
│       └── prompt_caching.py  # Multi-provider prompt optimization
├── data/
│   ├── raw-human-games/   # Human campaign data
│   └── llm-games/         # LLM simulation outputs
│       ├── metadata_index.json  # Index for fast filtering
│       ├── game-logs/     # Campaign JSON files with metadata
│       ├── summaries/     # Turn summary text files
│       └── scratchpads/   # Reasoning logs (if enabled)
└── tutorials/             # Jupyter notebook examples
```

## 📊 Analysis Features

- **Basic Metrics**: Time intervals, post lengths, player engagement
- **Semantic Analysis**: SBERT embeddings, semantic distance, session novelty, DSI
- **Jaccard Cohesion**: Lexical similarity between players within sessions
- **Multi-Campaign Comparisons**: Statistical aggregation with intelligent caching
- **Session-Based Analysis**: Configurable session boundaries for temporal analysis

## 🤖 Simulation Features

- **Character Extraction**: Generate personalities and backstories from human campaigns
- **Multi-Excerpt Extraction**: Smart context windows for late-joining characters
- **LLM Character Agents**: Memory-aware characters with realistic posting patterns
- **Turn Summarization**: Handle long games (10k+ turns) with automatic summarization
- **Multi-Provider Support**: Anthropic, OpenAI, Google with provider-specific caching
- **Scratchpad Reasoning**: Chain-of-thought for more realistic character responses
- **Prompt Optimization**: Hierarchical caching system for cost reduction
- **Save & Filter**: Save game logs with metadata, filter by model/settings on load

## 🛠️ Installation

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn

# For semantic analysis
pip install sentence-transformers torch transformers

# For LLM simulation
pip install litellm anthropic
```

## 📚 Documentation

- `tutorials/` - Interactive Jupyter notebook examples
- `CLAUDE.md` - Development guidelines for LLM assistance
- Function docstrings - Detailed parameter and return information

## 📄 License

MIT License - See LICENSE file for details