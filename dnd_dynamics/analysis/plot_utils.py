"""Helper plotting functions for D&D campaign analysis."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .metrics.result import MetricResult


# ===================================================================
# HISTOGRAM PLOTTING
# ===================================================================


def plot_histogram(data,
                   colors=None,
                   edgecolor='none',
                   alpha=0.5,
                   colormap='viridis',
                   xlabel=None,
                   bins=None,
                   log_bins=False,
                   log_y=False,
                   figsize=(4, 4),
                   ylabel="Counts",
                   labels=None):
    """
    Plot histogram(s) with colors evenly spaced across a colormap.

    Parameters
    ----------
    data : array-like or list of array-like
        Single array/list OR list of arrays/lists to plot as histograms
    colors : list, optional
        Specific colors (if None, uses colormap spacing)
    edgecolor : str
        Edge color for bars (default: 'none')
    alpha : float
        Transparency level (default: 0.5)
    colormap : str
        Colormap name (default: 'viridis')
    xlabel : str, optional
        X-axis label
    bins : int, optional
        Number of bins (default: None uses matplotlib default)
    log_bins : bool
        Use log-spaced bins and log x-axis (default: False)
    log_y : bool
        Use log scale for y-axis (default: False)
    figsize : tuple
        Figure size (default: (4, 4))
    ylabel : str
        Y-axis label (default: "Counts")
    labels : list, optional
        Labels for each dataset
    """
    plt.figure(figsize=figsize)

    # Auto-detect if data is a single array or list of arrays
    try:
        if isinstance(data[0], (list, np.ndarray)):
            data_list = data
        else:
            data_list = [data]
    except:
        data_list = [data]

    # Remove NaN values from all datasets
    clean_data_list = []
    for dataset in data_list:
        clean_dataset = np.array(dataset)[~np.isnan(np.array(dataset))]
        clean_data_list.append(clean_dataset)

    n_plots = len(clean_data_list)

    if colors is None:
        cmap = cm.get_cmap(colormap)
        if n_plots == 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(i / (n_plots - 1)) for i in range(n_plots)]

    # Calculate common bins across all datasets
    all_data = np.concatenate(clean_data_list)

    if len(all_data) == 0:
        raise ValueError("All data contains only NaN values")

    if log_bins:
        positive_data = all_data[all_data > 0]
        if len(positive_data) == 0:
            raise ValueError("log_bins=True requires positive values in data")
        min_val = np.min(positive_data)
        max_val = np.max(positive_data)
        if bins is None:
            bins = 50
        common_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
    else:
        if bins is None:
            bins = 50
        min_val = np.min(all_data)
        max_val = np.max(all_data)
        common_bins = np.linspace(min_val, max_val, bins)

    # Plot all histograms
    for i, dataset in enumerate(clean_data_list):
        label = labels[i] if labels is not None else None
        plt.hist(dataset,
                 bins=common_bins,
                 color=colors[i],
                 edgecolor=edgecolor,
                 alpha=alpha,
                 label=label)

    if log_bins:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    sns.despine()
    plt.ylabel(ylabel)

    if xlabel is not None:
        plt.xlabel(xlabel, clip_on=False)

    # Style axes
    ax = plt.gca()
    ax.tick_params(colors='#4a4a4a', width=0.5)
    ax.spines['left'].set_color('#4a4a4a')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#4a4a4a')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.xaxis.label.set_color('#4a4a4a')
    ax.yaxis.label.set_color('#4a4a4a')


def plot_comparison_histograms(data,
                               colors=None,
                               edgecolor='none',
                               alpha=0.5,
                               colormap='viridis',
                               xlabel=None,
                               bins=None,
                               log_bins=False,
                               log_y=False,
                               labels=None,
                               figsize=None,
                               ylabel="Counts"):
    """
    Plot comparison histograms with the first dataset appearing in every subplot,
    and each subsequent dataset compared against it in separate vertical subplots.

    Parameters
    ----------
    data : list
        List of arrays/lists (must have at least 2 datasets)
    colors : list, optional
        Specific colors (if None, uses colormap spacing)
    edgecolor : str
        Edge color for bars (default: 'none')
    alpha : float
        Transparency level (default: 0.5)
    colormap : str
        Colormap name (default: 'viridis')
    xlabel : str, optional
        X-axis label
    bins : int, optional
        Number of bins
    log_bins : bool
        Use log-spaced bins and log x-axis (default: False)
    log_y : bool
        Use log scale for y-axis (default: False)
    labels : list, optional
        Labels for each dataset
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    ylabel : str
        Y-axis label (default: "Counts")

    Returns
    -------
    fig, axes : tuple
        Figure and axes objects
    """
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("data must be a list with at least 2 datasets")

    # Remove NaN values from all datasets
    clean_data_list = []
    for dataset in data:
        clean_dataset = np.array(dataset)[~np.isnan(np.array(dataset))]
        clean_data_list.append(clean_dataset)

    if any(len(dataset) == 0 for dataset in clean_data_list):
        raise ValueError("One or more datasets contain only NaN values")

    n_datasets = len(clean_data_list)
    n_subplots = n_datasets - 1

    if figsize is None:
        figsize = (8, 3 * n_subplots)

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(hspace=0)

    if n_subplots == 1:
        axes = [axes]

    # Set up colors
    if colors is None:
        cmap = cm.get_cmap(colormap)
        colors = [cmap(i / (n_datasets - 1)) for i in range(n_datasets)]

    # Calculate common bins
    all_data = np.concatenate(clean_data_list)

    if log_bins:
        positive_data = all_data[all_data > 0]
        if len(positive_data) == 0:
            raise ValueError("log_bins=True requires positive values in data")
        min_val = np.min(positive_data)
        max_val = np.max(positive_data)
        if bins is None:
            bins = 50
        common_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
    else:
        if bins is None:
            bins = 50
        min_val = np.min(all_data)
        max_val = np.max(all_data)
        common_bins = np.linspace(min_val, max_val, bins)

    reference_data = clean_data_list[0]
    reference_color = colors[0]
    reference_label = labels[0] if labels is not None else "Reference"

    for i in range(n_subplots):
        ax = axes[i]
        comparison_data = clean_data_list[i + 1]
        comparison_color = colors[i + 1]
        comparison_label = labels[i + 1] if labels is not None else f"Dataset {i + 1}"

        ax.hist(reference_data, bins=common_bins, color=reference_color,
                edgecolor=edgecolor, alpha=alpha, label=reference_label)
        ax.hist(comparison_data, bins=common_bins, color=comparison_color,
                edgecolor=edgecolor, alpha=alpha, label=comparison_label)

        if log_bins:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        sns.despine(ax=ax)

        if i == n_subplots // 2:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")

        ax.tick_params(colors='#4a4a4a', width=0.5, length=2)
        ax.spines['left'].set_color('#4a4a4a')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('#4a4a4a')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.xaxis.label.set_color('#4a4a4a')
        ax.yaxis.label.set_color('#4a4a4a')
        ax.set_facecolor("none")
        ax.minorticks_off()

    if xlabel is not None:
        axes[-1].set_xlabel(xlabel)

    # Set same y-limits for all subplots
    max_ylim = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_ylim)
        ax.set_yticks([0, np.round(max_ylim * .7 / 10) * 10])

    return fig, axes


def plot_model_comparison_histograms(data_by_model,
                                      model_labels,
                                      colors=None,
                                      edgecolor='none',
                                      alpha=0.5,
                                      colormap='viridis',
                                      xlabel=None,
                                      bins=None,
                                      log_bins=False,
                                      log_y=False,
                                      figsize=None,
                                      ylabel="Counts"):
    """
    Plot comparison histograms for multiple models side-by-side.

    Each column shows one model's results, with rows comparing the reference
    category (first in each list) against other categories.

    Parameters
    ----------
    data_by_model : dict
        Dict mapping model_label -> list of arrays (one per category).
        Each list should have the same number of categories in the same order.
        First category in each list is the reference.
    model_labels : list
        List of model labels in display order (must match keys in data_by_model)
    colors : list, optional
        Specific colors for categories (if None, uses colormap spacing)
    edgecolor : str
        Edge color for bars (default: 'none')
    alpha : float
        Transparency level (default: 0.5)
    colormap : str
        Colormap name (default: 'viridis')
    xlabel : str, optional
        X-axis label (shown only on bottom row)
    bins : int, optional
        Number of bins
    log_bins : bool
        Use log-spaced bins and log x-axis (default: False)
    log_y : bool
        Use log scale for y-axis (default: False)
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    ylabel : str
        Y-axis label (default: "Counts")

    Returns
    -------
    fig, axes : tuple
        Figure and 2D array of axes objects (rows x cols)
    """
    n_models = len(model_labels)
    if n_models < 1:
        raise ValueError("Need at least 1 model")

    # Get number of categories from first model
    first_model_data = data_by_model[model_labels[0]]
    n_categories = len(first_model_data)
    if n_categories < 2:
        raise ValueError("Need at least 2 categories per model")

    n_rows = n_categories - 1  # Reference vs each other category

    # Clean data and validate structure
    clean_data = {}
    for label in model_labels:
        if label not in data_by_model:
            raise ValueError(f"Model '{label}' not found in data_by_model")
        model_data = data_by_model[label]
        if len(model_data) != n_categories:
            raise ValueError(f"Model '{label}' has {len(model_data)} categories, expected {n_categories}")
        clean_data[label] = [
            np.array(d)[~np.isnan(np.array(d))] for d in model_data
        ]

    if figsize is None:
        figsize = (2.5 * n_models, 0.6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_models, figsize=figsize, sharex='col', sharey='row')

    # Handle single row/column cases
    if n_rows == 1 and n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_models == 1:
        axes = axes.reshape(-1, 1)

    # Set up colors for categories
    if colors is None:
        cmap = cm.get_cmap(colormap)
        colors = [cmap(i / (n_categories - 1)) for i in range(n_categories)]

    # Calculate common bins across ALL data from ALL models
    all_data = np.concatenate([
        np.concatenate(clean_data[label]) for label in model_labels
        if len(np.concatenate(clean_data[label])) > 0
    ])

    if len(all_data) == 0:
        raise ValueError("All data contains only NaN values")

    if log_bins:
        positive_data = all_data[all_data > 0]
        if len(positive_data) == 0:
            raise ValueError("log_bins=True requires positive values in data")
        min_val = np.min(positive_data)
        max_val = np.max(positive_data)
        if bins is None:
            bins = 50
        common_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
    else:
        if bins is None:
            bins = 50
        min_val = np.min(all_data)
        max_val = np.max(all_data)
        common_bins = np.linspace(min_val, max_val, bins)

    # Plot each model as a column
    for col, label in enumerate(model_labels):
        model_clean = clean_data[label]
        reference_data = model_clean[0]
        reference_color = colors[0]

        for row in range(n_rows):
            ax = axes[row, col]
            comparison_data = model_clean[row + 1]
            comparison_color = colors[row + 1]

            ax.hist(reference_data, bins=common_bins, color=reference_color,
                    edgecolor=edgecolor, alpha=alpha)
            ax.hist(comparison_data, bins=common_bins, color=comparison_color,
                    edgecolor=edgecolor, alpha=alpha)

            if log_bins:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            sns.despine(ax=ax)

            ax.tick_params(colors='#4a4a4a', width=0.5, length=2)
            ax.spines['left'].set_color('#4a4a4a')
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('#4a4a4a')
            ax.spines['bottom'].set_linewidth(0.5)
            ax.xaxis.label.set_color('#4a4a4a')
            ax.yaxis.label.set_color('#4a4a4a')
            ax.set_facecolor("none")
            ax.minorticks_off()

        # Add model name as column title
        axes[0, col].set_title(label, fontsize=9)

        # Add xlabel only on bottom row
        if xlabel is not None:
            axes[-1, col].set_xlabel(xlabel if col == n_models // 2 else '')

    # Add ylabel only on leftmost column, middle row
    middle_row = n_rows // 2
    axes[middle_row, 0].set_ylabel(ylabel)

    # Set same y-limits across all subplots
    max_ylim = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(0, max_ylim)
        ax.set_yticks([0, np.round(max_ylim * .7 / 10) * 10])

    plt.subplots_adjust(hspace=0, wspace=0.1)

    return fig, axes


# ===================================================================
# CATEGORIZATION AND AGGREGATION
# ===================================================================


def categorize_campaigns(campaign_names: List[str],
                         category_fields: List[str],
                         metadata_index_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Categorize campaigns based on metadata fields.

    Args:
        campaign_names: List of campaign names
        category_fields: List of metadata fields to use for grouping
                        e.g., ['model', 'include_player_personalities']
        metadata_index_path: Path to metadata index JSON (defaults to standard location)

    Returns:
        Dict mapping category_name -> list of campaign names
        e.g., {'human': [...], 'model:gpt-4o, include_player_personalities:True': [...]}
    """
    if metadata_index_path is None:
        repo_root = Path(__file__).parent.parent.parent
        metadata_index_path = repo_root / 'data' / 'llm-games' / 'metadata_index.json'

    with open(metadata_index_path) as f:
        metadata_index = json.load(f)

    categories = {'human': []}

    for name in campaign_names:
        if name in metadata_index:
            metadata = metadata_index[name]
            parts = []
            for field in category_fields:
                value = metadata.get(field)
                if value is not None:
                    parts.append(f"{field}:{value}")
            category_key = ', '.join(parts) if parts else 'llm_other'
            if category_key not in categories:
                categories[category_key] = []
            categories[category_key].append(name)
        else:
            categories['human'].append(name)

    return categories


def aggregate_by_category(metric_data: List[np.ndarray],
                          campaign_names: List[str],
                          categories: Dict[str, List[str]],
                          category_order: Optional[List[str]] = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Aggregate metric data by category.

    Args:
        metric_data: List of metric arrays, one per campaign (in same order as campaign_names)
        campaign_names: List of campaign names corresponding to metric_data
        categories: Dict from categorize_campaigns()
        category_order: Optional list specifying order of categories in output

    Returns:
        Tuple of (aggregated_data, category_order) where aggregated_data is a list of
        concatenated arrays, one per category
    """
    name_to_idx = {name: i for i, name in enumerate(campaign_names)}

    if category_order is None:
        category_order = ['human'] + sorted(k for k in categories.keys() if k != 'human')

    result = []
    for cat in category_order:
        cat_names = categories.get(cat, [])
        if cat_names:
            cat_data = [metric_data[name_to_idx[n]] for n in cat_names if n in name_to_idx]
            result.append(np.concatenate(cat_data) if cat_data else np.array([]))
        else:
            result.append(np.array([]))

    return result, category_order


def aggregate_metric(results: Dict[str, MetricResult],
                     campaign_names: List[str],
                     categories: Dict[str, List[str]],
                     category_order: List[str],
                     metric_name: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Aggregate any metric (series or summary) across campaigns by category.
    Auto-detects whether metric_name is in series or summary.

    Args:
        results: Dict mapping campaign_name -> MetricResult
        campaign_names: List of campaign names to process
        categories: Dict mapping category_name -> list of campaign names
        category_order: List of categories in display order
        metric_name: Name of metric to aggregate (checks series first, then summary)

    Returns:
        Tuple of (aggregated_values, category_order) where aggregated_values[i]
        is the concatenated array for category_order[i]
    """
    values = []
    for name in campaign_names:
        if name not in results or results[name] is None:
            values.append(np.array([]))
            continue

        result = results[name]
        if metric_name in result.series:
            values.append(result.series[metric_name])
        elif metric_name in result.summary:
            # Wrap scalar in array for aggregation
            values.append(np.array([result.summary[metric_name]]))
        else:
            values.append(np.array([]))

    return aggregate_by_category(values, campaign_names, categories, category_order)


# ===================================================================
# TIMELINE AND ANALYSIS PLOTS
# ===================================================================

def plot_distance_timeline(df: pd.DataFrame, 
                          distance_col: str = "semantic_distance_w1",
                          date_col: str = "date",
                          rolling_window: int = 10) -> None:
    """
    Plot semantic distance over time with rolling average.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with distance and date columns
    distance_col : str
        Column containing semantic distances
    date_col : str
        Column containing timestamps
    rolling_window : int
        Window size for rolling average
    """
    import matplotlib.pyplot as plt
    
    # Calculate rolling average
    rolling_dist = df[distance_col].rolling(window=rolling_window, center=True).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[distance_col], alpha=0.3, label='Raw distance')
    plt.plot(df[date_col], rolling_dist, linewidth=2, label=f'{rolling_window}-post rolling average')
    plt.xlabel('Date')
    plt.ylabel('Semantic Distance')
    plt.title('Semantic Distance Timeline')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_topic_timeline(df: pd.DataFrame,
                       topic_col: str = "topic",
                       date_col: str = "date") -> None:
    """
    Plot topic evolution over time as colored timeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with topic and date columns
    topic_col : str
        Column containing topic assignments
    date_col : str
        Column containing timestamps
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create daily topic counts
    df_plot = df.copy()
    df_plot['date_day'] = pd.to_datetime(df_plot[date_col]).dt.date
    
    topic_counts = df_plot.groupby(['date_day', topic_col]).size().unstack(fill_value=0)
    
    # Plot stacked bar chart
    plt.figure(figsize=(15, 8))
    topic_counts.plot(kind='bar', stacked=True, 
                     colormap='tab20', figsize=(15, 8))
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.title('Topic Evolution Over Time')
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_session_novelty(session_stats: pd.DataFrame) -> None:
    """
    Plot session novelty statistics.
    
    Parameters
    ----------
    session_stats : pd.DataFrame
        Output from session_novelty() function
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mean distance distribution
    axes[0, 0].hist(session_stats['mean_distance'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Mean Semantic Distance')
    axes[0, 0].set_xlabel('Distance')
    
    # Max distance distribution
    axes[0, 1].hist(session_stats['max_distance'], bins=20, alpha=0.7)
    axes[0, 1].set_title('Max Semantic Distance')
    axes[0, 1].set_xlabel('Distance')
    
    # Post count vs mean distance
    axes[1, 0].scatter(session_stats['post_count'], session_stats['mean_distance'], alpha=0.6)
    axes[1, 0].set_xlabel('Posts per Session')
    axes[1, 0].set_ylabel('Mean Distance')
    axes[1, 0].set_title('Session Size vs Novelty')
    
    # Standard deviation
    axes[1, 1].hist(session_stats['std_distance'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Distance Std Deviation')
    axes[1, 1].set_xlabel('Std Distance')
    
    plt.tight_layout()
    plt.show()

def plot_topic_transitions(transition_matrix: pd.DataFrame,
                          top_n: int = 10) -> None:
    """
    Plot topic transition matrix as heatmap.
    
    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Output from topic_transition_matrix()
    top_n : int
        Number of top topics to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Select top N most common topics
    topic_counts = transition_matrix.sum(axis=1).sort_values(ascending=False)
    top_topics = topic_counts.head(top_n).index
    
    # Subset matrix
    subset_matrix = transition_matrix.loc[top_topics, top_topics]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(subset_matrix, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f'Topic Transition Matrix (Top {top_n} Topics)')
    plt.xlabel('Next Topic')
    plt.ylabel('Current Topic')
    plt.tight_layout()
    plt.show()

