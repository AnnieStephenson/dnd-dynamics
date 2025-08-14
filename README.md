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
import analysis.data_loading as dl
import analysis.creativity_metrics as creativity

# Load campaigns and analyze creativity
campaigns = dl.load_campaigns('human', max_campaigns=10)  
results = creativity.analyze_creativity(campaigns)
```

### Simulation Example  
```python
import llm_scaffolding.dnd_simulation as sim

# Extract parameters from human campaign
params = sim.extract_campaign_parameters('path/to/campaign.json')

# Create LLM characters and run simulation
characters = sim.create_characters(params)
session = sim.GameSession(characters, 'campaign_name', scratchpad=True)

# Run the simulation using extracted turn sequence
session.run_scenario(params['initial_scenario'], params['character_turns'])
```

## 📁 Project Structure

```
dnd-dynamics/
├── analysis/              # Campaign analysis tools
│   ├── data_loading.py    # Load and process campaign data  
│   ├── basic_metrics.py   # Time intervals, post lengths, engagement
│   ├── creativity_metrics.py # Semantic analysis, topic modeling
│   ├── cohesion_metrics.py   # Linguistic alignment analysis
│   └── batch.py          # Multi-campaign processing & caching
├── llm_scaffolding/       # LLM simulation system
│   ├── dnd_simulation.py  # Character agents & game sessions
│   ├── prompt_caching.py  # Multi-provider prompt optimization  
│   └── api_config.py      # LLM provider configuration
└── tutorials/             # Jupyter notebook examples
```

## 📊 Analysis Features

- **Basic Metrics**: Time intervals, post lengths, player engagement
- **Creativity Analysis**: Semantic embeddings, topic modeling, novelty detection  
- **Cohesion Analysis**: Linguistic alignment between players using lexical similarity
- **Multi-Campaign Comparisons**: Statistical aggregation with 2500x caching speedup
- **Session-Based Analysis**: Configurable session boundaries for temporal analysis

## 🤖 Simulation Features

- **Character Extraction**: Generate personalities and backstories from human campaigns
- **LLM Character Agents**: Memory-aware characters with realistic posting patterns
- **Multi-Provider Support**: Anthropic, OpenAI, Google with provider-specific caching
- **Scratchpad Reasoning**: Chain-of-thought for more realistic character responses
- **Prompt Optimization**: Hierarchical caching system for cost reduction

## 🛠️ Installation

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn

# For creativity analysis
pip install sentence-transformers bertopic torch

# For LLM simulation  
pip install litellm anthropic

# For cohesion analysis
pip install align nltk
```

## 📈 Performance

| Feature | Memory | Time | Cache Benefit |
|---------|--------|------|---------------|
| Basic Analysis (100 campaigns) | ~2GB | 15-30 min | 2500x speedup |
| Creativity Analysis (50 campaigns) | ~4GB | 1-2 hours | Embeddings cached |
| LLM Simulation (100 turns) | ~500MB | 10-20 min | 90% cost reduction |

## 📚 Documentation

- `tutorials/` - Interactive Jupyter notebook examples
- `CLAUDE.md` - Development guidelines for LLM assistance
- Function docstrings - Detailed parameter and return information

## 📄 License

MIT License - See LICENSE file for details