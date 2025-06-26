# D&D Gameplay Log Analysis 🎲

A comprehensive Python toolkit for analyzing Dungeons & Dragons gameplay logs using statistical methods and advanced NLP techniques. This project provides both basic gameplay metrics and sophisticated creativity analysis across single campaigns or large multi-campaign datasets.

## 🎯 Project Overview

### What This Project Does
This toolkit analyzes D&D gameplay logs stored in JSON format to extract meaningful insights about:
- **Player engagement patterns** - Who posts when and how much
- **Narrative progression** - Time intervals, cumulative activity, message types
- **Semantic creativity** - How language and topics evolve during gameplay
- **Cross-campaign comparisons** - Statistical analysis across hundreds of campaigns

### Main Capabilities
- ⚡ **Basic Metrics Analysis**: Time intervals, post lengths, player statistics, character mentions
- 🧠 **Creativity Analysis**: Semantic embeddings, topic modeling, novelty detection
- 📊 **Multi-Campaign Comparisons**: Statistical aggregation and visualization across datasets
- 🚀 **Intelligent Caching**: 2500x speedup with incremental processing for large datasets

### Target Audience & Use Cases
- **Game Masters**: Understand player engagement and narrative patterns
- **Researchers**: Study online collaborative storytelling and creativity
- **Data Scientists**: Example of multi-modal analysis (statistical + NLP)
- **D&D Communities**: Compare campaign styles and identify successful patterns

---

## 📁 Project Architecture

```
dnd-dynamics/
├── dnd_analysis.py                      # Core single-campaign basic metrics
├── creative_metrics.py                  # Advanced NLP creativity analysis
├── tutorial.ipynb                       # Single-campaign basic walkthrough
├── tutorial_creativity.ipynb            # Single-campaign creativity analysis
├── multi_campaign_tutorial.ipynb       # Multi-campaign basic metrics
├── creativity_comparison_tutorial.ipynb # Multi-campaign creativity comparison
├── requirements.txt                     # Python dependencies
├── Game-Data/
│   └── data-labels.json                # Input campaign data
├── campaign_stats_cache/               # Cached analysis results
│   ├── basic_stats_N_campaigns.pkl    # Basic metrics cache
│   └── creativity_analysis_N_campaigns.pkl # Creativity cache
└── Plots/                             # Generated visualizations
```

### Core Components

#### 🔧 **Analysis Modules**
- **`dnd_analysis.py`**: Single-campaign statistical analysis functions
  - Time interval analysis, player statistics, character mentions
  - Functions: `load_dnd_data()`, `analyze_time_intervals()`, `analyze_post_lengths()`
  
- **`creative_metrics.py`**: Advanced NLP creativity analysis
  - Semantic embeddings, topic modeling, novelty scoring
  - Functions: `get_embeddings()`, `semantic_distance()`, `topic_model()`

#### 📓 **Tutorial Notebooks**
- **`tutorial.ipynb`**: Basic single-campaign analysis with visualizations
- **`tutorial_creativity.ipynb`**: Creativity analysis for one campaign
- **`multi_campaign_tutorial.ipynb`**: Compare basic metrics across campaigns
- **`creativity_comparison_tutorial.ipynb`**: Advanced creativity comparisons

#### 📊 **Data Files**
- **`Game-Data/data-labels.json`**: Input data in nested JSON format
- **`campaign_stats_cache/`**: Intelligent caching for fast re-analysis

---

## 🚀 Intelligent Caching System

### Cache File Naming Conventions

```bash
campaign_stats_cache/
├── basic_stats_5_campaigns.pkl          # Basic metrics for 5 campaigns
├── basic_stats_50_campaigns.pkl         # Basic metrics for 50 campaigns  
├── creativity_analysis_5_campaigns.pkl  # Creativity metrics for 5 campaigns
└── creativity_analysis_50_campaigns.pkl # Creativity metrics for 50 campaigns
```

### Incremental Processing Logic

The caching system provides **2500x speedup** through intelligent incremental computation:

1. **Cache Discovery**: Finds largest existing cache ≤ requested campaigns
2. **Gap Analysis**: Determines which additional campaigns need processing
3. **Incremental Computation**: Processes only new campaigns
4. **Result Merging**: Combines cached and new results efficiently
5. **Cache Update**: Saves complete result set for future use

### When to Use Force Refresh

```python
# Use force_refresh=True when:
analyze_campaigns(max_campaigns=10, force_refresh=True)  # Data has changed
analyze_campaigns(max_campaigns=10, force_refresh=True)  # Testing new code
analyze_campaigns(max_campaigns=10, force_refresh=True)  # Cache corruption
```

### Cache Validation

- **File integrity**: Handles corrupted pickle files gracefully
- **Data freshness**: Compares source data modification times
- **Version compatibility**: Graceful fallback if cache format changes

---

## 📋 Results Dictionary Structure

### Basic Metrics Results Structure

```python
# Single campaign result
campaign_result = {
    'time_intervals': {
        'interval_hours': [2.5, 1.2, 3.7, ...],           # Hours between posts
        'avg_interval_hours': 2.8,                         # Mean interval
        'interval_distribution': pd.DataFrame(...)          # Binned intervals
    },
    'cumulative_posts': {
        'daily_counts': [5, 12, 8, ...],                   # Posts per day
        'cumulative_counts': [5, 17, 25, ...],             # Running total
        'dates': ['2023-01-01', '2023-01-02', ...]         # Date strings
    },
    'post_lengths': {
        'word_counts_data': pd.DataFrame(...),              # Per-player word counts
        'top_players': ['Alice', 'Bob', 'Charlie'],         # Most active players
        'distribution_data': pd.DataFrame(...)              # Length distributions
    },
    'player_engagement': {
        'posts_per_player': {'Alice': 45, 'Bob': 32, ...}, # Post counts
        'avg_length_per_player': {'Alice': 25.3, ...},     # Avg words per post
        'top_n_players': ['Alice', 'Bob', 'Charlie']        # Most active
    },
    'character_mentions': {
        'total_mentions': 127,                              # Total name mentions
        'unique_characters_mentioned': 8,                   # Unique names
        'top_mentions': {'Gandalf': 15, 'Aragorn': 12},    # Most mentioned
        'full_counts': pd.Series(...)                       # Complete counts
    }
}

# Multi-campaign results
multi_campaign_results = {
    'campaign_id_1': campaign_result,                       # Individual results
    'campaign_id_2': campaign_result,
    ...
}
```

### Creativity Metrics Results Structure

```python
# Single campaign creativity result
creativity_result = {
    'embeddings': np.ndarray,                               # Sentence embeddings (768-dim)
    'semantic_distances': [0.12, 0.34, 0.08, ...],         # Message-to-message distances
    'session_novelty': {
        'novelty_scores': [0.65, 0.23, 0.89, ...],         # Per-session novelty
        'session_boundaries': [0, 15, 28, ...],             # Session start indices
        'avg_novelty': 0.59                                 # Mean novelty score
    },
    'topic_model': {
        'topics': pd.Series([2, 1, 3, 2, ...]),            # Topic assignments per message
        'model': BERTopic(...)                              # Fitted topic model object
    },
    'topic_transitions': pd.DataFrame(...),                 # Topic transition matrix
    'topic_change_rate': {
        'overall_rate': 0.25,                               # Overall change frequency
        'series': pd.Series([0, 1, 0, 1, ...])             # Per-message changes
    },
    'metadata': {
        'campaign_id': 'campaign_123',                      # Campaign identifier
        'total_messages': 156,                              # Message count
        'unique_players': 4,                                # Player count
        'date_range': {
            'start': '2023-01-15T10:30:00',                # First message timestamp
            'end': '2023-03-22T18:45:00'                   # Last message timestamp
        }
    }
}
```

### Aggregated Results Format

```python
# Cross-campaign aggregation
aggregated_results = {
    'campaign_summaries': {
        'campaign_1': {
            'total_messages': 156,                          # Campaign size
            'unique_players': 4,                            # Player count
            'avg_semantic_distance': 0.234,                # Mean creativity
            'avg_novelty_score': 0.591,                    # Mean novelty
            'topic_change_rate': 0.25                      # Topic dynamics
        }
    },
    'cross_campaign_stats': {
        'semantic_distance': {
            'mean': 0.198, 'std': 0.067,                   # Population statistics
            'min': 0.089, 'max': 0.345,                    # Range
            'campaigns_analyzed': 50                        # Sample size
        }
    },
    'distributions': {
        'semantic_distances': [0.198, 0.234, ...],         # Raw data for plotting
        'campaign_ids': ['camp_1', 'camp_2', ...],         # Campaign identifiers
        'campaign_sizes': [156, 203, 89, ...]              # Message counts
    },
    'summary': {
        'total_campaigns_analyzed': 50,                     # Overall count
        'total_messages_across_all_campaigns': 8947        # Total dataset size
    }
}
```

### Key Design Decisions

#### Why Nested Dictionaries?
1. **Flexible Access**: `results['campaign_123']['semantic_distances']` is intuitive
2. **Extensible**: Easy to add new metrics without breaking existing code
3. **Human-Readable**: Keys have clear semantic meaning
4. **JSON Serializable**: Can save/load results in multiple formats
5. **Campaign Lookup**: Direct access by campaign ID without searching

#### Key Naming Conventions
- **Snake_case**: Consistent with Python conventions (`semantic_distances`)
- **Descriptive Names**: `avg_interval_hours` vs `ai_h` for clarity
- **Plural for Lists**: `semantic_distances` (list), `avg_semantic_distance` (scalar)
- **Metadata Separation**: `metadata` key contains campaign info, not analysis results

#### Data Types Within Results
- **Lists**: Time series data, individual measurements
- **DataFrames**: Structured data for easy analysis/plotting
- **Scalars**: Summary statistics, aggregated values
- **Objects**: Fitted models (BERTopic, etc.) for further analysis

---

## 🚀 Getting Started Examples

### 1. Quick Start - Single Campaign Analysis

```python
import json
from dnd_analysis import load_dnd_data, analyze_time_intervals, analyze_post_lengths

# Load your data
with open('Game-Data/data-labels.json', 'r') as f:
    data = json.load(f)

# Pick one campaign
campaign_id = list(data.keys())[0]
single_campaign = {campaign_id: data[campaign_id]}

# Convert to DataFrame
df = load_dnd_data(single_campaign)
print(f"Loaded {len(df)} messages from {df['player'].nunique()} players")

# Analyze time intervals
intervals = analyze_time_intervals(df)
print(f"Average time between posts: {intervals['avg_interval_hours']:.2f} hours")

# Analyze post lengths
lengths = analyze_post_lengths(df)
print(f"Top player: {lengths['top_players'][0]} with {lengths['word_counts_data'].iloc[0]['total_words']} words")
```

**Expected Output:**
```
Loaded 89 messages from 4 players
Average time between posts: 2.34 hours
Top player: Alice with 432 words
```

### 2. Multi-Campaign Comparison with Caching

```python
from dnd_analysis import load_or_compute_incremental

# Start small - analyze 10 campaigns (builds cache)
print("=== Analyzing 10 campaigns ===")
results_10 = load_or_compute_incremental(
    max_campaigns=10,
    show_progress=True
)
print(f"Analyzed {len(results_10)} campaigns")

# Scale up - analyze 30 campaigns (uses cache + incremental)
print("\n=== Scaling to 30 campaigns ===")
results_30 = load_or_compute_incremental(
    max_campaigns=30,
    show_progress=True  # Will show: "Found cache with 10 campaigns, computing 20 additional"
)

# Access results by campaign ID
campaign_id = list(results_30.keys())[0]
campaign_data = results_30[campaign_id]
print(f"\nCampaign {campaign_id}:")
print(f"  Messages: {campaign_data['basic_stats']['metadata']['total_messages']}")
print(f"  Players: {campaign_data['basic_stats']['metadata']['unique_players']}")
print(f"  Avg interval: {campaign_data['time_intervals']['avg_interval_hours']:.2f}h")
```

### 3. Creativity Analysis Workflow

```python
from creative_metrics import analyze_creativity_all_campaigns, aggregate_creativity_metrics
import matplotlib.pyplot as plt

# Analyze creativity across 5 campaigns
creativity_results = analyze_creativity_all_campaigns(
    max_campaigns=5,
    cache_dir='creativity_cache',
    show_progress=True
)

# Aggregate for comparison
aggregated = aggregate_creativity_metrics(creativity_results)

# Navigate the results
for campaign_id, result in creativity_results.items():
    if result and 'semantic_distances' in result:
        distances = result['semantic_distances']
        avg_creativity = sum(d for d in distances if not pd.isna(d)) / len([d for d in distances if not pd.isna(d)])
        print(f"{campaign_id[:30]}... avg creativity: {avg_creativity:.4f}")

# Generate comparison plot
plt.figure(figsize=(10, 6))
for i, (campaign_id, result) in enumerate(creativity_results.items()):
    if result and 'semantic_distances' in result:
        distances = [d for d in result['semantic_distances'] if not pd.isna(d)]
        plt.hist(distances, alpha=0.5, label=campaign_id[:20], bins=15)
plt.xlabel('Semantic Distance')
plt.ylabel('Frequency')
plt.legend()
plt.title('Creativity Distribution Across Campaigns')
plt.show()
```

### 4. Cache Management Best Practices

```python
import os
import glob

# Check what's cached
cache_files = glob.glob('campaign_stats_cache/*.pkl')
print("Cached files:")
for f in sorted(cache_files):
    size = os.path.getsize(f) / 1024 / 1024  # MB
    print(f"  {os.path.basename(f)}: {size:.1f} MB")

# Development: Force refresh for testing
dev_results = load_or_compute_incremental(
    max_campaigns=5,
    force_refresh=True,  # Ignore cache, recompute everything
    show_progress=True
)

# Production: Let caching work
prod_results = load_or_compute_incremental(
    max_campaigns=100,   # Large dataset
    force_refresh=False, # Use cache when possible
    show_progress=True
)

# Clean up old cache files (optional)
import shutil
# shutil.rmtree('campaign_stats_cache')  # Nuclear option
# os.makedirs('campaign_stats_cache', exist_ok=True)
```

---

## ⚙️ Technical Requirements

### Python Environment
```bash
# Minimum Requirements
Python >= 3.8

# Core Dependencies
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 seaborn>=0.11.0

# NLP Dependencies (for creativity analysis)
pip install sentence-transformers>=2.2.0 bertopic>=0.15.0 torch>=1.12.0

# Complete Installation
pip install -r requirements.txt
```

### Data Format Expectations

Your `Game-Data/data-labels.json` should follow this structure:
```json
{
  "campaign_id_1": {
    "message_id_1": {
      "date": "2023-01-15T10:30:00",
      "player": "Alice", 
      "character": "Elara the Ranger",
      "text": "I draw my bow and aim at the orc",
      "in_combat": true,
      "actions": {"check": {"roll": "1d20+5"}},
      "name_mentions": ["Gandalf", "Thorin"]
    }
  }
}
```

Alternative format with paragraphs:
```json
{
  "campaign_id_1": {
    "message_id_1": {
      "paragraphs": {
        "0": {"text": "First paragraph"},
        "1": {"text": "Second paragraph"}
      }
    }
  }
}
```

### Memory Considerations

| Dataset Size | RAM Required | Processing Time | Cache Size |
|--------------|-------------|-----------------|------------|
| 10 campaigns | ~500 MB | 2-5 minutes | ~50 MB |
| 100 campaigns | ~2 GB | 15-30 minutes | ~200 MB |
| 500 campaigns | ~8 GB | 1-2 hours | ~800 MB |

### Performance Tips

1. **Start Small**: Test with 5-10 campaigns before scaling up
2. **Use Caching**: Never use `force_refresh=True` in production
3. **Incremental Processing**: Build up dataset size gradually (10 → 50 → 100)
4. **Creativity Analysis**: Most memory-intensive due to embeddings
5. **Batch Processing**: For >500 campaigns, consider processing in batches

---

## 🔄 Common Workflows

### Development Cycle
```python
# 1. Test code with small dataset
results = analyze_campaigns(max_campaigns=3, force_refresh=True)

# 2. Validate on medium dataset  
results = analyze_campaigns(max_campaigns=20, force_refresh=False)

# 3. Production run with full dataset
results = analyze_campaigns(max_campaigns=None, force_refresh=False)
```

### Production Analysis
```python
# Daily analysis workflow
def daily_analysis():
    # Basic metrics for trending
    basic = load_or_compute_incremental(max_campaigns=500)
    
    # Creativity deep-dive on subset
    creative = analyze_creativity_all_campaigns(max_campaigns=50)
    
    # Generate reports
    generate_daily_report(basic, creative)
    
    return basic, creative
```

### Troubleshooting Common Issues

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `ImportError: sentence-transformers` | Missing NLP dependencies | `pip install sentence-transformers` |
| `MemoryError` | Dataset too large | Reduce `max_campaigns` or add RAM |
| `KeyError: 'text'` | Data format mismatch | Check JSON structure |
| Cache corruption | Interrupted processing | Delete cache files, restart |
| `list indices must be integers` | Function argument error | Update to latest code version |

---

## 📚 API Reference

### Core Analysis Functions (`dnd_analysis.py`)

```python
def load_dnd_data(json_data: Dict) -> pd.DataFrame
    """Convert nested JSON to clean DataFrame"""

def analyze_time_intervals(df: pd.DataFrame) -> Dict
    """Calculate time between posts and patterns"""

def analyze_post_lengths(df: pd.DataFrame, include_all_players: bool = True) -> Dict  
    """Analyze word counts and length distributions"""

def analyze_player_engagement(df: pd.DataFrame) -> Dict
    """Player activity statistics and rankings"""

def load_or_compute_incremental(max_campaigns: int, **kwargs) -> Dict
    """Smart caching with incremental processing"""
```

### Creativity Functions (`creative_metrics.py`)

```python
def get_embeddings(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray
    """Generate sentence embeddings for text analysis"""

def semantic_distance(df: pd.DataFrame, embeddings: np.ndarray = None) -> pd.Series
    """Calculate semantic distances between consecutive messages"""

def topic_model(df: pd.DataFrame, n_topics: int = 20) -> Tuple[pd.Series, object]
    """Extract topics using BERTopic or LDA"""

def analyze_creativity_all_campaigns(max_campaigns: int, **kwargs) -> Dict
    """Multi-campaign creativity analysis with caching"""
```

### Key Parameters

- **`max_campaigns`**: Limit analysis to first N campaigns (None = all)
- **`force_refresh`**: Ignore cache and recompute (default: False) 
- **`show_progress`**: Display progress bars (default: True)
- **`cache_dir`**: Directory for cached results (default: 'campaign_stats_cache')

---

## 🤝 Contributing & Usage Tips

### Best Practices
- Always test with small datasets first (`max_campaigns=5`)
- Use descriptive variable names when accessing nested results
- Cache results locally for interactive analysis
- Include error handling for missing data/dependencies

### Performance Optimization
- Leverage the caching system - avoid `force_refresh=True`
- Process campaigns incrementally (10 → 50 → 100)
- Monitor memory usage with large datasets
- Use `show_progress=True` for long-running operations

### Extending the Analysis
The modular design makes it easy to add new metrics:
1. Add functions to `dnd_analysis.py` or `creative_metrics.py`
2. Update the multi-campaign processing loops
3. Add visualization code to the tutorial notebooks
4. Update this README with new capabilities

---

## 🎮 Data Processing and Turn Structure

The D&D analysis toolkit is specifically designed to handle the unique structural elements of tabletop RPG gameplay data, which differs significantly from standard text analysis. Understanding how the code processes these D&D-specific components is crucial for accurate analysis.

### Turn-Based Gameplay Structure

D&D gameplay follows a distinctive turn-based format where each "message" represents a complete player turn that may contain multiple components:

```json
{
  "campaign_id": {
    "message_123": {
      "date": "2023-01-15T10:30:00",
      "player": "Alice",
      "character": "Elara the Ranger",
      "paragraphs": {
        "0": {"text": "I carefully approach the ancient door"},
        "1": {"text": "**Out of Character:** Should I check for traps first?"},
        "2": {"text": "I draw my bow and ready an arrow"}
      },
      "actions": ["Stealth Check", "Perception Check", "Ready Action"],
      "in_combat": true,
      "name_mentions": ["Gandalf", "Ancient Door"]
    }
  }
}
```

### Numbered Paragraphs in Turns

**How It Works:**
The code processes turns containing multiple numbered paragraphs or sections within a single message through the `load_dnd_data()` function:

```python
# Paragraph processing logic
if 'paragraphs' in message_data:
    paragraphs = message_data.get('paragraphs', {})
    text_segments = []
    for para_id in sorted(paragraphs.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        para_data = paragraphs[para_id]
        if isinstance(para_data, dict) and 'text' in para_data:
            text_segments.append(para_data['text'])
    text_content = ' '.join(text_segments)
```

**Why This Matters:**
- **Narrative Flow**: Preserves the logical sequence of actions and thoughts within a turn
- **Context Preservation**: Maintains relationships between related paragraphs
- **Word Count Accuracy**: Ensures proper aggregation of text length metrics

### Actions List Format

**Critical Format Detail:**
Actions are stored as **lists in square brackets `[]`**, NOT as dictionaries in curly braces `{}`:

```json
// CORRECT FORMAT
"actions": ["Stealth Check", "Attack Roll", "Damage Roll"]

// INCORRECT FORMAT (not supported)
"actions": {"check": "stealth", "roll": "1d20+5"}
```

**Processing Logic:**
```python
# Actions extraction in load_dnd_data()
actions = message_data.get('actions', [])  # Expects list format

# Dice roll detection considers actions
def _detect_dice_rolls(row):
    # Check for dice patterns in actions list
    if isinstance(row['actions'], list):
        for action in row['actions']:
            if 'roll' in action.lower() or 'check' in action.lower():
                return True
    return False
```

### Character Labels and Content Types

**In-Character vs Out-of-Character Distinction:**

The code identifies different content types within turns:

```json
{
  "paragraphs": {
    "0": {"text": "I cast Fireball at the orc!"},           // In-character action
    "1": {"text": "**OOC:** Is that within range?"},        // Out-of-character question
    "2": {"text": "*Rolls 8d6 fire damage*"}                // Mechanical narration
  }
}
```

**Detection Methods:**
- **Markdown indicators**: `**OOC:**`, `**Out of Character:**`
- **Italics for mechanics**: `*rolls dice*`, `*takes damage*`
- **Character voice**: Direct speech without meta-indicators

**Why This Matters:**
- **Creativity Analysis**: In-character content shows narrative creativity vs. mechanical coordination
- **Player Engagement**: Separates roleplay from rule discussions
- **Topic Modeling**: Different content types cluster into different topics

### Turn Parsing Logic

**Multi-Component Extraction:**

```python
def load_dnd_data(json_data):
    # 1. Extract text from paragraphs (preserving order)
    if 'paragraphs' in message_data:
        text_segments = []
        for para_id in sorted(paragraphs.keys()):
            text_segments.append(para_data['text'])
        text_content = ' '.join(text_segments)
    
    # 2. Process actions separately
    actions = message_data.get('actions', [])
    
    # 3. Extract character mentions
    name_mentions = message_data.get('name_mentions', [])
    
    # 4. Determine message type
    message_type = _classify_message_type(combined_data)
```

**Message Type Classification:**
```python
def _classify_message_type(row):
    text = str(row['text']).lower()
    actions = row['actions']
    
    if row['in_combat']:
        return 'combat'
    elif isinstance(actions, list) and len(actions) > 0:
        return 'action'
    elif '"' in text or 'says' in text:
        return 'dialogue'
    else:
        return 'narrative'
```

### Impact on Analysis Metrics

**Word Count Analysis:**
- **Total Words**: Aggregates all paragraph text, excluding action labels
- **Content Type Weighting**: Different analysis for dialogue vs. narration vs. mechanics

**Creativity Metrics:**
- **Semantic Distance**: Calculated on combined paragraph text to preserve narrative flow
- **Topic Modeling**: In-character content weighted higher for creativity analysis
- **Novelty Scoring**: Actions and mechanics filtered out to focus on narrative creativity

**Player Engagement:**
- **Turn Complexity**: Measures based on paragraph count + action count
- **Roleplay vs. Mechanics**: Ratio of in-character to out-of-character content

### Technical Implementation Notes

**Data Extraction Methods:**

```python
# Robust paragraph extraction
def extract_paragraphs(message_data):
    if 'text' in message_data:
        # Legacy format: single text field
        return message_data['text']
    elif 'paragraphs' in message_data:
        # Modern format: structured paragraphs
        return combine_paragraphs(message_data['paragraphs'])
    else:
        return ""  # Handle missing text gracefully

# Action list validation
def validate_actions(actions):
    if isinstance(actions, list):
        return actions
    elif isinstance(actions, dict):
        # Convert legacy dict format to list
        return [f"{k}: {v}" for k, v in actions.items()]
    else:
        return []
```

**Missing Data Handling:**
- **Empty Paragraphs**: `if isinstance(para_data, dict) and 'text' in para_data` validation
- **Malformed Actions**: Graceful conversion from dict to list format when possible
- **Missing Timestamps**: NaT handling in time interval analysis
- **Incomplete Turns**: Partial data still processed for available metrics

**Why Structure Preservation Matters:**
1. **Accuracy**: D&D gameplay has specific patterns that generic text analysis misses
2. **Context**: Turn structure provides important narrative and mechanical context
3. **Comparability**: Standardized processing allows meaningful cross-campaign comparisons
4. **Extensibility**: Structured approach enables future D&D-specific metrics

This specialized processing ensures that the analysis accurately reflects the unique nature of collaborative storytelling in tabletop RPGs, rather than treating D&D sessions as generic chat logs.

---

Ready to analyze your D&D campaigns? Start with the `tutorial.ipynb` notebook and explore the rich world of gameplay analytics! 🎲✨