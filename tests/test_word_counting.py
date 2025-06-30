"""
Test suite for word counting functions in dnd_analysis.py

This test suite verifies that all word counting functions correctly include 
text from all paragraphs in posts, ensuring no content is accidentally excluded.
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dnd_analysis import *
import json
import pandas as pd
import numpy as np


@pytest.fixture
def sample_campaign_data():
    """Create sample campaign data with known paragraph structures for testing."""
    return {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Alice",
                "character": "Elara",
                "paragraphs": {
                    "0": {
                        "text": "I draw my bow carefully.",  # 5 words
                        "label": "in-character"
                    },
                    "1": {
                        "text": "Can I make a stealth check?",  # 6 words  
                        "label": "out-of-character"
                    },
                    "2": {
                        "text": "Rolling 1d20 plus three for stealth.",  # 6 words
                        "label": "mixed"
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            },
            "2": {
                "date": "2023-01-01T10:15:00", 
                "player": "Bob",
                "character": "Thorin",
                "paragraphs": {
                    "0": {
                        "text": "The dwarf charges forward with his axe raised high above his head.",  # 12 words
                        "label": "in-character"
                    },
                    "1": {
                        "text": "I attack the orc!",  # 4 words
                        "label": "in-character"
                    }
                },
                "actions": ["Attack"],
                "name_mentions": ["Thorin"],
                "in_combat": True
            },
            "3": {
                "date": "2023-01-01T10:30:00",
                "player": "Charlie",
                "character": "Gandalf", 
                "paragraphs": {
                    "0": {
                        "text": "",  # 0 words (empty paragraph)
                        "label": "in-character"
                    },
                    "1": {
                        "text": "You shall not pass!",  # 4 words
                        "label": "in-character"
                    },
                    "2": {
                        "text": "",  # 0 words (empty paragraph)
                        "label": "out-of-character"
                    },
                    "3": {
                        "text": "I cast fireball at level three.",  # 6 words
                        "label": "mixed"
                    }
                },
                "actions": ["Cast Spell"],
                "name_mentions": [],
                "in_combat": True
            },
            "4": {
                "date": "2023-01-01T11:00:00",
                "player": "David",
                "character": "Legolas",
                "text": "Single text field without paragraphs structure.",  # 6 words (old format)
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }


@pytest.fixture
def multi_paragraph_post():
    """Create a test post with multiple paragraphs and known word count."""
    return {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T12:00:00",
                "player": "TestPlayer",
                "character": "TestCharacter",
                "paragraphs": {
                    "0": {
                        "text": "First paragraph has exactly five words.",  # 6 words
                        "label": "in-character"
                    },
                    "1": {
                        "text": "Second paragraph contains seven distinct word tokens here.",  # 8 words
                        "label": "in-character"
                    },
                    "2": {
                        "text": "Third and final paragraph has exactly six words.",  # 8 words
                        "label": "out-of-character"
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }


@pytest.fixture
def edge_case_data():
    """Create test data with edge cases for word counting."""
    return {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T13:00:00",
                "player": "EdgePlayer",
                "character": "EdgeCharacter",
                "paragraphs": {
                    "0": {
                        "text": "Word with-hyphen and word's apostrophe.",  # 5 words
                        "label": "in-character"
                    },
                    "1": {
                        "text": "Multiple    spaces     between    words.",  # 4 words
                        "label": "in-character"
                    },
                    "2": {
                        "text": "Numbers 123 and symbols !@# count too.",  # 7 words
                        "label": "mixed"
                    },
                    "3": {
                        "text": "   Leading and trailing spaces   ",  # 4 words
                        "label": "out-of-character"
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            },
            "2": {
                "date": "2023-01-01T13:15:00",
                "player": "PunctuationPlayer", 
                "character": "PunctuationCharacter",
                "paragraphs": {
                    "0": {
                        "text": "!@#$%^&*()",  # 1 word (just punctuation)
                        "label": "out-of-character"
                    },
                    "1": {
                        "text": "",  # 0 words (empty)
                        "label": "in-character"
                    },
                    "2": {
                        "text": "Normal words after punctuation.",  # 4 words
                        "label": "in-character"
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }


def test_all_paragraphs_included_in_word_count(sample_campaign_data):
    """Test that word counting functions include text from all paragraphs in a post."""
    df = load_dnd_data(sample_campaign_data)
    
    # Test message 1: 3 paragraphs with 5+6+6=17 total words
    message_1 = df[df['message_id'] == 1].iloc[0]
    expected_total_words = 5 + 6 + 6  # "I draw my bow carefully." + "Can I make a stealth check?" + "Rolling 1d20 plus three for stealth."
    
    assert message_1['word_count'] == expected_total_words, \
        f"Message 1 total word count should be {expected_total_words}, got {message_1['word_count']}"
    
    # Verify in-character words (5 words from paragraph 0)
    assert message_1['in_character_word_count'] == 5, \
        f"Message 1 in-character word count should be 5, got {message_1['in_character_word_count']}"
    
    # Verify out-of-character words (6 words from paragraph 1)
    assert message_1['out_of_character_word_count'] == 6, \
        f"Message 1 out-of-character word count should be 6, got {message_1['out_of_character_word_count']}"
    
    # Verify mixed words (6 words from paragraph 2)
    assert message_1['mixed_word_count'] == 6, \
        f"Message 1 mixed word count should be 6, got {message_1['mixed_word_count']}"
    
    # Test message 2: 2 paragraphs with 12+4=16 total words
    message_2 = df[df['message_id'] == 2].iloc[0]
    expected_total_words = 12 + 4  # "The dwarf charges forward with his axe raised high above his head." + "I attack the orc!"
    
    assert message_2['word_count'] == expected_total_words, \
        f"Message 2 total word count should be {expected_total_words}, got {message_2['word_count']}"
    
    # All words should be in-character
    assert message_2['in_character_word_count'] == expected_total_words, \
        f"Message 2 in-character word count should be {expected_total_words}, got {message_2['in_character_word_count']}"


def test_paragraph_separation_preserved(multi_paragraph_post):
    """Test that paragraph boundaries don't affect word counting."""
    df = load_dnd_data(multi_paragraph_post)
    message = df.iloc[0]
    
    # Expected: 6 + 8 + 8 = 22 total words
    expected_total = 6 + 8 + 8
    
    assert message['word_count'] == expected_total, \
        f"Multi-paragraph post should have {expected_total} words, got {message['word_count']}"
    
    # Verify that the combined text is properly space-separated
    combined_text = message['text']
    manual_word_count = len(combined_text.split())
    
    assert manual_word_count == expected_total, \
        f"Manual word count should match calculated count: expected {expected_total}, got {manual_word_count}"
    
    # Verify no extra spaces were introduced at paragraph boundaries
    assert '  ' not in combined_text.replace('   ', '  '), \
        f"No double spaces should exist from paragraph joining: '{combined_text}'"


def test_in_character_vs_out_character_counting(sample_campaign_data):
    """Test that word counting handles character labels correctly."""
    df = load_dnd_data(sample_campaign_data)
    
    # Message 3 has mixed content: empty + 4 in-character + empty + 6 mixed = 10 total
    message_3 = df[df['message_id'] == 3].iloc[0]
    
    assert message_3['word_count'] == 10, \
        f"Message 3 should have 10 total words, got {message_3['word_count']}"
    
    assert message_3['in_character_word_count'] == 4, \
        f"Message 3 should have 4 in-character words, got {message_3['in_character_word_count']}"
    
    assert message_3['out_of_character_word_count'] == 0, \
        f"Message 3 should have 0 out-of-character words, got {message_3['out_of_character_word_count']}"
    
    assert message_3['mixed_word_count'] == 6, \
        f"Message 3 should have 6 mixed words, got {message_3['mixed_word_count']}"
    
    # Verify that sum of labeled words equals total (accounting for unlabeled)
    total_labeled = message_3['in_character_word_count'] + message_3['out_of_character_word_count'] + message_3['mixed_word_count']
    assert total_labeled == message_3['word_count'], \
        f"Sum of labeled words ({total_labeled}) should equal total word count ({message_3['word_count']})"


def test_word_count_functions_consistency(sample_campaign_data):
    """Test that all word-counting functions give consistent results."""
    df = load_dnd_data(sample_campaign_data)
    
    # Test analyze_post_lengths function
    post_lengths_overall = analyze_post_lengths(df, by_player=False)
    post_lengths_by_player = analyze_post_lengths(df, by_player=True)
    
    # Verify that analyze_post_lengths uses the same word_count column
    assert 'overall' in post_lengths_overall
    assert post_lengths_overall['overall']['count'] == len(df)
    
    # Test that by_player analysis includes all players
    unique_players = df['player'].unique()
    for player in unique_players:
        assert player in post_lengths_by_player, f"Player {player} should be in by_player analysis"
        
        player_df = df[df['player'] == player]
        expected_count = len(player_df)
        actual_count = post_lengths_by_player[player]['count']
        
        assert actual_count == expected_count, \
            f"Player {player} should have {expected_count} posts, got {actual_count}"
    
    # Test label-aware analysis consistency
    post_lengths_by_label = analyze_post_lengths_by_label(df, by_player=False)
    
    # Verify summary statistics match
    assert post_lengths_by_label['summary']['total_messages'] == len(df)


@pytest.mark.parametrize("paragraph_count", [1, 2, 3, 5, 10])
def test_multiple_paragraph_counts(paragraph_count):
    """Test word counting with varying numbers of paragraphs."""
    # Create test data with specified number of paragraphs
    paragraphs = {}
    expected_words = 0
    
    for i in range(paragraph_count):
        # Each paragraph has i+1 words: "word", "word word", "word word word", etc.
        para_text = " ".join(["word"] * (i + 1))
        paragraphs[str(i)] = {
            "text": para_text,
            "label": "in-character" if i % 2 == 0 else "out-of-character"
        }
        expected_words += i + 1
    
    test_data = {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "TestPlayer",
                "character": "TestCharacter",
                "paragraphs": paragraphs,
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }
    
    df = load_dnd_data(test_data)
    message = df.iloc[0]
    
    assert message['word_count'] == expected_words, \
        f"Message with {paragraph_count} paragraphs should have {expected_words} words, got {message['word_count']}"


def test_edge_cases(edge_case_data):
    """Test edge cases for word counting."""
    df = load_dnd_data(edge_case_data)
    
    # Message 1: Test special characters and spacing
    message_1 = df[df['message_id'] == 1].iloc[0]
    
    # Expected: 5 + 4 + 7 + 4 = 20 words
    expected_words = 5 + 4 + 7 + 4
    
    assert message_1['word_count'] == expected_words, \
        f"Edge case message 1 should have {expected_words} words, got {message_1['word_count']}"
    
    # Message 2: Test punctuation and empty paragraphs
    message_2 = df[df['message_id'] == 2].iloc[0]
    
    # Expected: 1 + 0 + 4 = 5 words
    expected_words = 1 + 0 + 4
    
    assert message_2['word_count'] == expected_words, \
        f"Edge case message 2 should have {expected_words} words, got {message_2['word_count']}"


def test_empty_paragraphs_handling(sample_campaign_data):
    """Test that empty paragraphs are handled correctly and don't affect word counts."""
    df = load_dnd_data(sample_campaign_data)
    
    # Message 3 has empty paragraphs (indexes 0 and 2)
    message_3 = df[df['message_id'] == 3].iloc[0]
    
    # Should only count non-empty paragraphs: "You shall not pass!" (4) + "I cast fireball at level three." (6) = 10
    assert message_3['word_count'] == 10, \
        f"Message with empty paragraphs should have 10 words, got {message_3['word_count']}"
    
    # Note: The current implementation does create spaces from empty paragraphs
    # This is expected behavior - empty paragraphs contribute spaces when joined
    # The important thing is that word count is still correct (10 words)
    combined_text = message_3['text']
    
    # Verify that despite extra spaces, word count is still correct
    manual_word_count = len(combined_text.split())
    assert manual_word_count == 10, \
        f"Manual word count should be 10 despite spaces: got {manual_word_count} from '{combined_text}'"


def test_old_format_compatibility(sample_campaign_data):
    """Test that old format (single text field) still works correctly."""
    df = load_dnd_data(sample_campaign_data)
    
    # Message 4 uses old format with single text field
    message_4 = df[df['message_id'] == 4].iloc[0]
    
    # "Single text field without paragraphs structure." = 6 words
    assert message_4['word_count'] == 6, \
        f"Old format message should have 6 words, got {message_4['word_count']}"
    
    # Old format should be marked as unlabeled
    assert message_4['primary_label'] == 'unlabeled', \
        f"Old format should have primary_label 'unlabeled', got '{message_4['primary_label']}'"


def test_aggregation_functions_use_all_words():
    """Test that aggregation functions include all word counts from all campaigns."""
    # Create test data with known word counts
    test_data = {
        "campaign-1": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Player1",
                "character": "Char1",
                "text": "Five words in this message.",  # 5 words
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        },
        "campaign-2": {
            "1": {
                "date": "2023-01-01T11:00:00", 
                "player": "Player2",
                "character": "Char2",
                "text": "This message has exactly seven words total.",  # 7 words
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }
    
    # Load campaigns separately
    campaign_1_df = load_dnd_data({"campaign-1": test_data["campaign-1"]})
    campaign_2_df = load_dnd_data({"campaign-2": test_data["campaign-2"]})
    
    campaign_dataframes = {
        "campaign-1": campaign_1_df,
        "campaign-2": campaign_2_df
    }
    
    # Run analysis with original JSON data
    all_results = analyze_all_campaigns(campaign_dataframes, test_data, show_progress=False)
    
    # Check aggregated results
    aggregated = all_results['aggregated']
    
    assert 'post_lengths' in aggregated
    
    # Should have total count of 2 messages
    assert aggregated['post_lengths']['count'] == 2, \
        f"Aggregated analysis should include 2 messages, got {aggregated['post_lengths']['count']}"
    
    # Word counts should be preserved in aggregation
    all_word_counts = aggregated['post_lengths']['word_counts_data']
    expected_word_counts = sorted([5, 7])  # From our test messages
    actual_word_counts = sorted(all_word_counts)
    
    assert actual_word_counts == expected_word_counts, \
        f"Aggregated word counts should be {expected_word_counts}, got {actual_word_counts}"


def test_consistency_across_analysis_functions():
    """Test that different analysis functions report consistent word count statistics."""
    # Use the sample data
    test_data = {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Alice",
                "character": "Elara",
                "paragraphs": {
                    "0": {"text": "First paragraph with five words.", "label": "in-character"},  # 5 words
                    "1": {"text": "Second paragraph has six words total.", "label": "out-of-character"}  # 6 words
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }
    
    df = load_dnd_data(test_data)
    
    # Test different analysis functions
    post_lengths = analyze_post_lengths(df, by_player=False)
    post_lengths_by_label = analyze_post_lengths_by_label(df, by_player=False)
    summary_report = generate_summary_report(df)
    
    # All should report the same total word count (11 words)
    expected_words = 11
    
    # Check raw data
    assert df.iloc[0]['word_count'] == expected_words, \
        f"Raw data should show {expected_words} words"
    
    # Check post_lengths analysis
    assert post_lengths['overall']['max_words'] == expected_words, \
        f"Post lengths analysis should show max {expected_words} words"
    
    # Check summary report
    assert summary_report['posting_patterns']['longest_post_words'] == expected_words, \
        f"Summary report should show longest post {expected_words} words"
    
    # Verify label-specific counts add up to total
    # Note: Labels use underscores in results, not hyphens
    label_results = post_lengths_by_label
    in_char_total = label_results['in_character']['overall']['total_words']
    out_char_total = label_results['out_of_character']['overall']['total_words']
    
    assert in_char_total + out_char_total == expected_words, \
        f"Label-specific word counts should sum to total: {in_char_total} + {out_char_total} != {expected_words}"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__])