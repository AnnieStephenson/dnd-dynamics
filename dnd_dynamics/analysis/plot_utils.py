 # Helper plotting functions

import pandas as pd
import numpy as np

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

