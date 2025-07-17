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
- 📋 **Paragraph-Level Action Analysis**: Track action types (spells, weapons, dialogue, rolls) and character labels at paragraph granularity
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
├── dnd_analysis.py                       # Core single-campaign basic metrics
├── interaction_analysis.py              # Advanced NLP interaction analysis
├── tutorial.ipynb                       # Single-campaign basic walkthrough
├── multi_campaign_tutorial.ipynb        # Multi-campaign basic metrics
├── interaction_comparison_tutorial.ipynb # Multi-campaign interaction comparison
├── requirements.txt                      # Python dependencies
├── Game-Data/
│   └── data-labels.json                 # Input campaign data
├── campaign_stats_cache/                # Cached analysis results
│   ├── basic_stats_N_campaigns.pkl     # Basic metrics cache
│   └── creativity_analysis_N_campaigns.pkl # Interaction cache
└── Plots/                              # Generated visualizations
```

### Core Components

#### 🔧 **Analysis Modules**
- **`dnd_analysis.py`**: Single-campaign statistical analysis functions
  - Time interval analysis, player statistics, character mentions, player participation
  - **Paragraph-level action analysis**: Analyze action types (spells, weapons, dialogue, etc.) and character labels at paragraph level
  - Functions: `load_dnd_data()`, `analyze_time_intervals()`, `analyze_post_lengths()`, `analyze_paragraph_actions()`, `calculate_player_campaign_participation()`
  
- **`interaction_analysis.py`**: Advanced NLP interaction analysis
  - Semantic embeddings, topic modeling, novelty scoring
  - Functions: `get_embeddings()`, `semantic_distance()`, `topic_model()`

#### 📓 **Tutorial Notebooks**
- **`tutorial.ipynb`**: Basic single-campaign analysis with visualizations
- **`multi_campaign_tutorial.ipynb`**: Compare basic metrics across campaigns
- **`interaction_comparison_tutorial.ipynb`**: Advanced interaction analysis comparisons

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
├── creativity_analysis_5_campaigns.pkl  # Interaction metrics for 5 campaigns
└── creativity_analysis_50_campaigns.pkl # Interaction metrics for 50 campaigns
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

## 📋 Paragraph-Level Action Analysis

### Overview

The paragraph-level action analysis feature provides granular insights into D&D gameplay by analyzing individual paragraphs within messages. This goes beyond simple message classification to understand specific action types and character voice at the paragraph level.

### Features

- **Action Type Detection**: Identifies paragraphs containing spells, weapons, dialogue, dice rolls, name mentions
- **Character Label Analysis**: Tracks in-character vs out-of-character content at paragraph granularity
- **Multi-Campaign Aggregation**: Combines paragraph-level statistics across entire datasets
- **Backward Compatibility**: Works alongside existing message-level analysis functions

### Usage

```python
import dnd_analysis as dnd

# Load campaign data with paragraph structure
json_data = load_campaign_json('Game-Data/data-labels.json')

# Analyze paragraph actions for single campaign
results = dnd.analyze_paragraph_actions(json_data)

# Multi-campaign analysis with paragraph actions
campaign_dfs, json_data = dnd.load_all_campaigns('Game-Data/data-labels.json', return_json=True)
all_results = dnd.analyze_all_campaigns(campaign_dfs, json_data)

# Access paragraph action results
paragraph_stats = all_results['aggregated']['paragraph_actions']
print(f"Spells cast: {paragraph_stats['spells_paragraphs']}")
print(f"In-character content: {paragraph_stats['in_character_percentage']:.1f}%")
```

### Action Types Tracked

- **`name_mentions`**: Paragraphs referencing character names
- **`spells`**: Paragraphs involving spell casting
- **`dialogue`**: Paragraphs containing character speech
- **`roll`**: Paragraphs with dice rolling mechanics
- **`weapon`**: Paragraphs involving weapon use
- **`no_action`**: Paragraphs with no specific action type

### Character Labels Tracked

- **`in-character`**: Content written from character perspective
- **`out-of-character`**: Meta-commentary and player discussion
- **`mixed`**: Paragraphs containing both in-character and OOC content
- **`unlabeled`**: Paragraphs without label information

### Requirements

- Requires JSON data with paragraph structure (not just processed DataFrames)
- Compatible with both single-campaign and multi-campaign analysis workflows
- Gracefully handles missing or malformed paragraph data

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
    'post_lengths_by_label': {                              # NEW: Label-aware analysis
        'in_character': {
            'overall': {'mean_words': 62.1, 'count': 120, 'total_words': 7452}
        },
        'out_of_character': {
            'overall': {'mean_words': 25.3, 'count': 89, 'total_words': 2252}
        },
        'mixed': {
            'overall': {'mean_words': 45.7, 'count': 34, 'total_words': 1554}
        },
        'summary': {
            'total_messages': 243,
            'messages_with_in_character': 120,
            'messages_with_out_of_character': 89,
            'messages_with_mixed': 34
        }
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
    },
    'paragraph_actions': {                                  # NEW: Paragraph-level action analysis
        'name_mentions_paragraphs': 45,                    # Paragraphs with name mentions
        'spells_paragraphs': 32,                           # Paragraphs with spell casting
        'dialogue_paragraphs': 78,                         # Paragraphs with dialogue
        'roll_paragraphs': 23,                             # Paragraphs with dice rolls
        'weapon_paragraphs': 19,                           # Paragraphs with weapon use
        'no_action_paragraphs': 156,                       # Paragraphs with no specific action
        'total_paragraphs': 353,                           # Total paragraphs analyzed
        'in_character_paragraphs': 267,                    # In-character paragraphs
        'out_of_character_paragraphs': 52,                 # Out-of-character paragraphs
        'mixed_paragraphs': 28,                            # Mixed content paragraphs
        'unlabeled_paragraphs': 6                          # Unlabeled paragraphs
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
    'embeddings': np.ndarray,                               # Combined sentence embeddings (768-dim)
    'semantic_distances': [0.12, 0.34, 0.08, ...],         # Combined message-to-message distances
    'label_embeddings': {                                   # NEW: Label-specific embeddings
        'in-character': np.ndarray,                         # In-character embeddings only
        'out-of-character': np.ndarray,                     # Out-of-character embeddings only
        'mixed': np.ndarray                                 # Mixed content embeddings only
    },
    'label_semantic_distances': {                           # NEW: Label-specific distances
        'in-character': [0.15, 0.28, 0.09, ...],          # In-character semantic distances
        'out-of-character': [0.08, 0.41, 0.12, ...],      # Out-of-character semantic distances
        'mixed': [0.22, 0.19, 0.31, ...]                  # Mixed content semantic distances
    },
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
            'avg_semantic_distance': 0.234,                # Mean creativity (combined)
            'avg_novelty_score': 0.591,                    # Mean novelty
            'topic_change_rate': 0.25,                     # Topic dynamics
            'in_character_semantic_distance': 0.289,       # NEW: In-character creativity
            'out_of_character_semantic_distance': 0.156,   # NEW: Out-of-character creativity
            'mixed_semantic_distance': 0.201,              # NEW: Mixed content creativity
            'in_character_message_count': 78,              # NEW: In-character message count
            'out_of_character_message_count': 64,          # NEW: Out-of-character message count
            'mixed_message_count': 14                      # NEW: Mixed message count
        }
    },
    'cross_campaign_stats': {
        'semantic_distance': {
            'mean': 0.198, 'std': 0.067,                   # Population statistics (combined)
            'min': 0.089, 'max': 0.345,                    # Range
            'campaigns_analyzed': 50                        # Sample size
        },
        'in_character_semantic_distance': {                # NEW: In-character cross-campaign stats
            'mean': 0.267, 'std': 0.082,                   # In-character population statistics
            'min': 0.124, 'max': 0.412,                    # In-character range
            'campaigns_analyzed': 47,                       # Campaigns with in-character content
            'total_messages_analyzed': 3542                 # Total in-character messages
        },
        'out_of_character_semantic_distance': {            # NEW: Out-of-character cross-campaign stats
            'mean': 0.145, 'std': 0.053,                   # Out-of-character population statistics
            'min': 0.067, 'max': 0.289,                    # Out-of-character range
            'campaigns_analyzed': 49,                       # Campaigns with out-of-character content
            'total_messages_analyzed': 4103                 # Total out-of-character messages
        },
        'mixed_semantic_distance': {                       # NEW: Mixed content cross-campaign stats
            'mean': 0.201, 'std': 0.071,                   # Mixed content population statistics
            'min': 0.098, 'max': 0.356,                    # Mixed content range
            'campaigns_analyzed': 32,                       # Campaigns with mixed content
            'total_messages_analyzed': 1302                 # Total mixed messages
        }
    },
    'distributions': {
        'semantic_distances': [0.198, 0.234, ...],         # Raw data for plotting (combined)
        'in_character_semantic_distances': [0.267, ...],   # NEW: In-character distribution data
        'out_of_character_semantic_distances': [0.145, ...], # NEW: Out-of-character distribution data
        'mixed_semantic_distances': [0.201, ...],          # NEW: Mixed content distribution data
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

### 3. Label-Aware Creativity Analysis Workflow

```python
from interaction_analysis import analyze_creativity_all_campaigns, aggregate_creativity_metrics
import matplotlib.pyplot as plt

# Analyze creativity across 5 campaigns with label-aware processing
creativity_results = analyze_creativity_all_campaigns(
    max_campaigns=5,
    cache_dir='creativity_cache',
    show_progress=True
)

# Aggregate for comparison (now includes label-aware metrics)
aggregated = aggregate_creativity_metrics(creativity_results)

# Navigate label-aware results
for campaign_id, result in creativity_results.items():
    print(f"\nCampaign: {campaign_id[:30]}...")
    
    # Combined creativity
    if result and 'semantic_distances' in result:
        distances = result['semantic_distances']
        avg_creativity = sum(d for d in distances if not pd.isna(d)) / len([d for d in distances if not pd.isna(d)])
        print(f"  Combined avg creativity: {avg_creativity:.4f}")
    
    # Label-specific creativity
    if result and 'label_semantic_distances' in result:
        for label, distances in result['label_semantic_distances'].items():
            if len(distances) > 0:
                avg_label_creativity = sum(d for d in distances if not pd.isna(d)) / len([d for d in distances if not pd.isna(d)])
                print(f"  {label} avg creativity: {avg_label_creativity:.4f} ({len(distances)} messages)")

# Generate label-aware comparison plot
plt.figure(figsize=(15, 10))

for i, (campaign_id, result) in enumerate(creativity_results.items()):
    if result and 'label_semantic_distances' in result:
        
        # Plot in-character creativity
        if 'in-character' in result['label_semantic_distances']:
            distances = [d for d in result['label_semantic_distances']['in-character'] if not pd.isna(d)]
            if distances:
                plt.subplot(2, 2, 1)
                plt.hist(distances, alpha=0.7, label=campaign_id[:15], bins=15)
                plt.title('In-Character Creativity Distribution')
                plt.xlabel('Semantic Distance')
                plt.ylabel('Frequency')
                
        # Plot out-of-character patterns
        if 'out-of-character' in result['label_semantic_distances']:
            distances = [d for d in result['label_semantic_distances']['out-of-character'] if not pd.isna(d)]
            if distances:
                plt.subplot(2, 2, 2)
                plt.hist(distances, alpha=0.7, label=campaign_id[:15], bins=15)
                plt.title('Out-of-Character Communication Patterns')
                plt.xlabel('Semantic Distance')
                plt.ylabel('Frequency')

plt.tight_layout()
plt.legend()
plt.show()

# Access label-specific cross-campaign statistics
print("\n=== LABEL-AWARE CROSS-CAMPAIGN STATISTICS ===")
for stat_name, stats in aggregated['cross_campaign_stats'].items():
    if 'semantic_distance' in stat_name:
        print(f"{stat_name}: mean={stats['mean']:.4f}, campaigns={stats['campaigns_analyzed']}")
```

### 4. Player Campaign Participation Analysis

```python
from dnd_analysis import load_or_compute_incremental, calculate_player_campaign_participation
import matplotlib.pyplot as plt
import numpy as np

# Load multi-campaign analysis results (using caching for efficiency)
all_results = load_or_compute_incremental(
    max_campaigns=100,
    show_progress=True
)

# Calculate how many campaigns each player has participated in
player_participation = calculate_player_campaign_participation(all_results)

# Display participation statistics
campaign_counts = list(player_participation.values())
print(f"📊 PLAYER PARTICIPATION ANALYSIS")
print(f"  Total unique players: {len(player_participation):,}")
print(f"  Average campaigns per player: {np.mean(campaign_counts):.2f}")
print(f"  Most active player: {max(campaign_counts)} campaigns")
print(f"  Players in only 1 campaign: {sum(1 for c in campaign_counts if c == 1):,}")

# Show top players
print(f"\n🏆 MOST ACTIVE PLAYERS:")
for i, (player, count) in enumerate(list(player_participation.items())[:5], 1):
    print(f"  {i}. {player[:30]}: {count} campaigns")

# Create participation distribution plot
plt.figure(figsize=(12, 8))
plt.hist(campaign_counts, bins=min(20, max(campaign_counts)), alpha=0.7, edgecolor='black')
plt.xlabel('Number of Campaigns Played')
plt.ylabel('Number of Players')
plt.title('Distribution of Campaign Participation Across Players')
plt.grid(True, alpha=0.3)

# Add summary statistics
mean_campaigns = np.mean(campaign_counts)
median_campaigns = np.median(campaign_counts)
plt.axvline(mean_campaigns, color='red', linestyle='--', label=f'Mean: {mean_campaigns:.1f}')
plt.axvline(median_campaigns, color='orange', linestyle='--', label=f'Median: {median_campaigns:.1f}')

plt.legend()
plt.tight_layout()
plt.show()

# Community insights
single_campaign = sum(1 for count in campaign_counts if count == 1)
multi_campaign = len(campaign_counts) - single_campaign
print(f"\n💡 COMMUNITY INSIGHTS:")
print(f"  One-time players: {single_campaign:,} ({single_campaign/len(campaign_counts)*100:.1f}%)")
print(f"  Multi-campaign players: {multi_campaign:,} ({multi_campaign/len(campaign_counts)*100:.1f}%)")
print(f"  Core community (5+ campaigns): {sum(1 for c in campaign_counts if c >= 5):,} players")
```

### 5. Cache Management Best Practices

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

## 🧪 Testing

The project includes a comprehensive test suite to verify that word counting functions correctly include all paragraphs from posts and handle various edge cases.

### Running Tests

```bash
# Install testing dependencies
pip install pytest>=7.0.0

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_word_counting.py

# Run tests with verbose output
pytest tests/ -v

# Run tests with coverage (if pytest-cov installed)
pytest tests/ --cov=dnd_analysis
```

### Test Coverage

The test suite covers:

- **Multi-paragraph word counting**: Verifies all paragraphs are included in total word counts
- **Label-aware counting**: Tests separate counting for in-character, out-of-character, and mixed content
- **Edge cases**: Empty paragraphs, special characters, punctuation, multiple spaces
- **Function consistency**: Ensures all word-counting functions produce consistent results
- **Aggregation accuracy**: Verifies multi-campaign aggregation preserves individual word counts
- **Old format compatibility**: Tests backward compatibility with single-text-field format

### Test Structure

```
tests/
├── test_word_counting.py        # Comprehensive word counting tests
└── __pycache__/                 # Compiled test files (auto-generated)
```

### Adding New Tests

When adding new word-counting functionality:

1. Create test data with known, manually-counted word totals
2. Test both new paragraph format and old single-text format
3. Verify consistency across all analysis functions
4. Include edge cases (empty content, special characters, etc.)
5. Test integration with existing caching and aggregation systems

### Test Philosophy

The test suite follows these principles:

- **Manual verification**: Expected word counts are manually calculated for test data
- **Comprehensive coverage**: Tests cover all word-counting functions and edge cases
- **Integration testing**: Verifies functions work together correctly in multi-campaign analysis
- **Clear assertions**: Each test includes descriptive error messages explaining what went wrong

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
1. Add functions to `dnd_analysis.py` or `interaction_analysis.py`
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

**Label-Based Content Processing:**

The data contains pre-labeled content at the paragraph level using the "label" key:

```json
{
  "paragraphs": {
    "0": {
      "text": "I cast Fireball at the orc!",
      "label": "in-character"
    },
    "1": {
      "text": "Is that within range?",
      "label": "out-of-character"
    },
    "2": {
      "text": "Rolling 8d6 fire damage: *rolls dice* 28 points of damage!",
      "label": "mixed"
    }
  }
}
```

**Label Types in Data:**
- **"in-character"**: Roleplay content, character actions, dialogue (52.0% of paragraphs)
- **"out-of-character"**: Meta-discussion, administrative content, OOC questions (43.4% of paragraphs)
- **"mixed"**: Combined narrative and mechanical content (4.6% of paragraphs)
- **Missing labels**: Small portion of unlabeled content (1.4% of paragraphs)

**Why Label-Aware Processing Matters:**
- **Creativity Analysis**: In-character content reveals pure narrative creativity vs. coordination
- **Player Engagement**: Separates roleplay intensity from administrative overhead
- **Topic Modeling**: Different content types naturally cluster into distinct topics
- **Accurate Metrics**: Post length, complexity, and semantic analysis vary significantly by content type

### Turn Parsing Logic

**Label-Aware Multi-Component Extraction:**

```python
def load_dnd_data(json_data):
    # 1. Extract text from paragraphs by label type
    if 'paragraphs' in message_data:
        all_text_segments = []
        in_char_segments = []
        out_char_segments = []
        mixed_segments = []
        
        for para_id in sorted(paragraphs.keys()):
            para_data = paragraphs[para_id]
            para_text = para_data['text']
            all_text_segments.append(para_text)
            
            # Separate by label
            para_label = para_data.get('label', 'unlabeled')
            if para_label == 'in-character':
                in_char_segments.append(para_text)
            elif para_label == 'out-of-character':
                out_char_segments.append(para_text)
            elif para_label == 'mixed':
                mixed_segments.append(para_text)
        
        # Create separate text fields for each label type
        text_content = ' '.join(all_text_segments)
        in_character_text = ' '.join(in_char_segments)
        out_of_character_text = ' '.join(out_char_segments)
        mixed_text = ' '.join(mixed_segments)
    
    # 2. Process actions separately (unchanged)
    actions = message_data.get('actions', [])
    
    # 3. Extract character mentions (unchanged)
    name_mentions = message_data.get('name_mentions', [])
    
    # 4. Determine primary label and message type
    primary_label = _determine_primary_label(label_counts)
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

**Label-Aware Word Count Analysis:**
- **Total Words**: Aggregates all paragraph text across all labels
- **Separated Counts**: `in_character_word_count`, `out_of_character_word_count`, `mixed_word_count`
- **Label-Specific Statistics**: Mean, median, distribution analysis for each content type

**Label-Aware Creativity Metrics:**
- **Separate Embeddings**: Generate embeddings for each label type independently
- **Label-Specific Semantic Distance**: Calculate creativity metrics separately for in-character vs. out-of-character content
- **Pure Narrative Analysis**: In-character content provides cleaner creativity measurements
- **Administrative Filtering**: Out-of-character content reveals coordination patterns

**Enhanced Player Engagement:**
- **Roleplay Intensity**: Ratio of in-character to total content per player
- **Administrative Overhead**: Out-of-character message frequency and length
- **Content Balance**: Mixed content indicates narrative/mechanical integration
- **Primary Label**: Dominant content type per message for classification

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