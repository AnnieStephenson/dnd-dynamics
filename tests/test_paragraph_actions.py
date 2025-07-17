"""
Test suite for paragraph-level action analysis functions in dnd_analysis.py

This test suite verifies that the analyze_paragraph_actions function correctly
processes paragraph-level action data and character labels according to the 
actual data structure.
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dnd_analysis import analyze_paragraph_actions, analyze_all_campaigns, load_dnd_data
import json
import pandas as pd
import numpy as np


@pytest.fixture
def sample_paragraph_data():
    """Create sample campaign data with paragraph-level actions for testing."""
    return {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Alice",
                "character": "Elara",
                "paragraphs": {
                    "0": {
                        "text": "I draw my bow carefully.",
                        "label": "in-character",
                        "actions": ["weapon"]
                    },
                    "1": {
                        "text": "Can I make a stealth check?",
                        "label": "out-of-character",
                        "actions": ["roll"]
                    },
                    "2": {
                        "text": "Elara looks around nervously.",
                        "label": "in-character",
                        "actions": ["name_mentions"]
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
                        "text": "I cast fireball!",
                        "label": "in-character",
                        "actions": ["spells"]
                    },
                    "1": {
                        "text": "Rolling for spell damage.",
                        "label": "mixed",
                        "actions": ["roll"]
                    },
                    "2": {
                        "text": "The spell illuminates the cavern.",
                        "label": "in-character",
                        "actions": []  # No specific action
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": True
            },
            "3": {
                "date": "2023-01-01T10:30:00",
                "player": "Charlie",
                "character": "Gandalf",
                "paragraphs": {
                    "0": {
                        "text": "What's everyone doing?",
                        "label": "out-of-character",
                        "actions": []
                    },
                    "1": {
                        "text": "Speaking to the group about strategy.",
                        "label": "in-character",
                        "actions": ["dialogue"]
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }


@pytest.fixture
def edge_case_paragraph_data():
    """Create edge case test data for paragraph actions."""
    return {
        "edge-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "EdgePlayer",
                "character": "EdgeCharacter",
                "paragraphs": {
                    "0": {
                        "text": "Multiple actions in one paragraph.",
                        "label": "in-character",
                        "actions": ["weapon", "dialogue", "name_mentions"]  # Multiple actions
                    },
                    "1": {
                        "text": "Empty actions list.",
                        "label": "unlabeled",
                        "actions": []  # Empty actions
                    },
                    "2": {
                        "text": "Missing actions field.",
                        "label": "mixed"
                        # No actions field at all
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            },
            "2": {
                "date": "2023-01-01T11:00:00",
                "player": "TestPlayer",
                "character": "TestChar",
                "paragraphs": {
                    "0": {
                        "text": "Unknown action type.",
                        "label": "in-character",
                        "actions": ["unknown_action", "spells"]  # Mix of known and unknown
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }


def test_basic_paragraph_actions_counting(sample_paragraph_data):
    """Test basic functionality of analyze_paragraph_actions."""
    results = analyze_paragraph_actions(sample_paragraph_data)
    
    # Test basic structure
    assert isinstance(results, dict)
    
    # Test action type counts
    assert results['weapon_paragraphs'] == 1  # Message 1, paragraph 0
    assert results['roll_paragraphs'] == 2   # Message 1 para 1, Message 2 para 1
    assert results['name_mentions_paragraphs'] == 1  # Message 1, paragraph 2
    assert results['spells_paragraphs'] == 1  # Message 2, paragraph 0
    assert results['dialogue_paragraphs'] == 1  # Message 3, paragraph 1
    assert results['no_action_paragraphs'] == 2  # Message 2 para 2, Message 3 para 0
    
    # Test total paragraphs
    assert results['total_paragraphs'] == 8  # 3 + 3 + 2 paragraphs
    
    # Test character label counts
    assert results['in_character_paragraphs'] == 5  # 2 + 2 + 1
    assert results['out_of_character_paragraphs'] == 2  # 1 + 0 + 1
    assert results['mixed_paragraphs'] == 1  # 0 + 1 + 0
    assert results['unlabeled_paragraphs'] == 0  # All paragraphs have labels


def test_character_label_distribution(sample_paragraph_data):
    """Test that character labels are counted correctly."""
    results = analyze_paragraph_actions(sample_paragraph_data)
    
    # Verify label totals add up
    total_labeled = (results['in_character_paragraphs'] + 
                    results['out_of_character_paragraphs'] + 
                    results['mixed_paragraphs'] + 
                    results['unlabeled_paragraphs'])
    
    assert total_labeled == results['total_paragraphs'], \
        f"Sum of labeled paragraphs ({total_labeled}) should equal total ({results['total_paragraphs']})"


def test_edge_cases_paragraph_actions(edge_case_paragraph_data):
    """Test edge cases like multiple actions, empty actions, and missing fields."""
    results = analyze_paragraph_actions(edge_case_paragraph_data)
    
    # Test handling of multiple actions in one paragraph
    # Paragraph with ["weapon", "dialogue", "name_mentions"] should increment all three
    assert results['weapon_paragraphs'] >= 1
    assert results['dialogue_paragraphs'] >= 1
    assert results['name_mentions_paragraphs'] >= 1
    
    # Test handling of empty actions and missing actions fields
    assert results['no_action_paragraphs'] >= 2  # At least 2 paragraphs with no actions
    
    # Test handling of unknown action types (should be ignored)
    assert results['spells_paragraphs'] >= 1  # Known action should still be counted
    
    # Test unlabeled content
    assert results['unlabeled_paragraphs'] >= 1  # At least one unlabeled paragraph
    
    # Total should be correct
    assert results['total_paragraphs'] == 5  # 3 + 1 paragraphs


def test_empty_campaign_data():
    """Test handling of empty or malformed campaign data."""
    # Empty data
    results = analyze_paragraph_actions({})
    assert results['total_paragraphs'] == 0
    assert results['name_mentions_paragraphs'] == 0
    
    # Campaign with no paragraphs
    no_paragraphs_data = {
        "campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Player",
                "character": "Character",
                "text": "Old format without paragraphs",
                "actions": []
            }
        }
    }
    
    results = analyze_paragraph_actions(no_paragraphs_data)
    assert results['total_paragraphs'] == 0


def test_multi_campaign_paragraph_analysis():
    """Test paragraph analysis across multiple campaigns."""
    multi_campaign_data = {
        "campaign-1": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Player1",
                "character": "Char1",
                "paragraphs": {
                    "0": {
                        "text": "First campaign action.",
                        "label": "in-character",
                        "actions": ["weapon"]
                    }
                }
            }
        },
        "campaign-2": {
            "1": {
                "date": "2023-01-01T11:00:00",
                "player": "Player2",
                "character": "Char2",
                "paragraphs": {
                    "0": {
                        "text": "Second campaign spell.",
                        "label": "in-character", 
                        "actions": ["spells"]
                    },
                    "1": {
                        "text": "OOC comment.",
                        "label": "out-of-character",
                        "actions": []
                    }
                }
            }
        }
    }
    
    results = analyze_paragraph_actions(multi_campaign_data)
    
    # Should count actions from both campaigns
    assert results['weapon_paragraphs'] == 1  # From campaign-1
    assert results['spells_paragraphs'] == 1  # From campaign-2
    assert results['no_action_paragraphs'] == 1  # OOC comment
    assert results['total_paragraphs'] == 3  # 1 + 2 paragraphs
    
    # Should count labels from both campaigns
    assert results['in_character_paragraphs'] == 2
    assert results['out_of_character_paragraphs'] == 1


def test_paragraph_actions_integration_with_multi_campaign():
    """Test that paragraph actions integrates properly with multi-campaign analysis."""
    # Create test data
    test_data = {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Alice",
                "character": "Elara",
                "paragraphs": {
                    "0": {
                        "text": "I attack with my sword.",
                        "label": "in-character",
                        "actions": ["weapon"]
                    },
                    "1": {
                        "text": "Rolling for attack.",
                        "label": "mixed",
                        "actions": ["roll"]
                    }
                },
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }
    
    # Load as dataframes
    campaign_dataframes = {
        "test-campaign": load_dnd_data(test_data)
    }
    
    # Run multi-campaign analysis with JSON data
    all_results = analyze_all_campaigns(
        campaign_dataframes=campaign_dataframes, 
        original_json_data=test_data,
        show_progress=False
    )
    
    # Check that paragraph actions are included in per-campaign results
    assert 'per_campaign' in all_results
    assert 'test-campaign' in all_results['per_campaign']
    assert 'paragraph_actions' in all_results['per_campaign']['test-campaign']
    
    # Check paragraph actions results
    para_results = all_results['per_campaign']['test-campaign']['paragraph_actions']
    assert para_results['weapon_paragraphs'] == 1
    assert para_results['roll_paragraphs'] == 1
    assert para_results['total_paragraphs'] == 2
    assert para_results['in_character_paragraphs'] == 1
    assert para_results['mixed_paragraphs'] == 1
    
    # Check that aggregated results include paragraph actions
    assert 'aggregated' in all_results
    if 'paragraph_actions' in all_results['aggregated']:
        agg_para = all_results['aggregated']['paragraph_actions']
        assert agg_para['weapon_paragraphs'] == 1
        assert agg_para['total_paragraphs'] == 2


def test_paragraph_actions_without_json_data():
    """Test that multi-campaign analysis works without JSON data (backwards compatibility)."""
    # Create test data
    test_data = {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Alice",
                "character": "Elara",
                "text": "Simple message without paragraph structure.",
                "actions": [],
                "name_mentions": [],
                "in_combat": False
            }
        }
    }
    
    # Load as dataframes only (no JSON data)
    campaign_dataframes = {
        "test-campaign": load_dnd_data(test_data)
    }
    
    # Run multi-campaign analysis without JSON data
    all_results = analyze_all_campaigns(
        campaign_dataframes=campaign_dataframes,
        original_json_data=None,
        show_progress=False
    )
    
    # Should work without paragraph actions
    assert 'per_campaign' in all_results
    assert 'test-campaign' in all_results['per_campaign']
    
    # Paragraph actions should not be present
    assert 'paragraph_actions' not in all_results['per_campaign']['test-campaign']
    
    # Aggregated results should not include paragraph actions
    assert 'aggregated' in all_results
    assert 'paragraph_actions' not in all_results['aggregated']


def test_action_type_edge_cases():
    """Test various action type edge cases and unknown types."""
    edge_data = {
        "test-campaign": {
            "1": {
                "date": "2023-01-01T10:00:00",
                "player": "Player",
                "character": "Character",
                "paragraphs": {
                    "0": {
                        "text": "Known actions.",
                        "label": "in-character",
                        "actions": ["weapon", "spells", "dialogue", "roll", "name_mentions"]
                    },
                    "1": {
                        "text": "Unknown actions.",
                        "label": "in-character",
                        "actions": ["unknown_action", "invalid_type", "custom_action"]
                    },
                    "2": {
                        "text": "Mixed known and unknown.",
                        "label": "in-character",
                        "actions": ["weapon", "unknown_action", "spells"]
                    },
                    "3": {
                        "text": "Non-list actions.",
                        "label": "in-character",
                        "actions": "not_a_list"  # Should be handled gracefully
                    }
                }
            }
        }
    }
    
    results = analyze_paragraph_actions(edge_data)
    
    # Known actions should be counted correctly
    assert results['weapon_paragraphs'] == 2  # Paragraphs 0 and 2
    assert results['spells_paragraphs'] == 2   # Paragraphs 0 and 2
    assert results['dialogue_paragraphs'] == 1  # Paragraph 0
    assert results['roll_paragraphs'] == 1     # Paragraph 0
    assert results['name_mentions_paragraphs'] == 1  # Paragraph 0
    
    # Unknown actions should not crash the system
    assert results['total_paragraphs'] == 4
    
    # At least one paragraph should be counted as no action (paragraph 3 with invalid actions)
    assert results['no_action_paragraphs'] >= 1


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__])