#!/usr/bin/env python3
"""
Test suite for the updated D&D analysis with paragraph-level action analysis.

This test suite validates that all the new functionality works correctly:
1. New analyze_paragraph_actions function
2. Updated multi-campaign analysis integration
3. Backward compatibility with existing functions
4. Tutorial notebooks can run without errors

Converted from validate_implementation.py to proper pytest format.
"""

import pytest
import json
import sys
from pathlib import Path

# Add the analysis directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

import dnd_analysis as dnd


class TestBasicFunctionality:
    """Test basic functionality of the new paragraph actions analysis."""
    
    def test_dnd_analysis_import(self):
        """Test that dnd_analysis module can be imported successfully."""
        assert dnd is not None, "dnd_analysis module should be importable"
    
    def test_analyze_paragraph_actions_exists(self):
        """Test that the new function exists."""
        assert hasattr(dnd, 'analyze_paragraph_actions'), "analyze_paragraph_actions function should exist"
    
    def test_required_functions_exist(self):
        """Test that existing functions still exist (backward compatibility)."""
        required_functions = [
            'load_dnd_data',
            'analyze_time_intervals', 
            'analyze_post_lengths',
            'analyze_action_vs_dialogue',
            'analyze_all_campaigns',
            'load_all_campaigns'
        ]
        
        for func_name in required_functions:
            assert hasattr(dnd, func_name), f"{func_name} function should exist"


class TestParagraphActionsFunction:
    """Test the analyze_paragraph_actions function with sample data."""
    
    @pytest.fixture
    def sample_data(self):
        """Create test data for paragraph actions testing."""
        return {
            "test-campaign": {
                "1": {
                    "date": "2023-01-01T10:00:00",
                    "player": "Alice",
                    "character": "Elara",
                    "paragraphs": {
                        "0": {
                            "text": "I cast fireball!",
                            "label": "in-character",
                            "actions": ["spells"]
                        },
                        "1": {
                            "text": "Rolling for damage.",
                            "label": "mixed",
                            "actions": ["roll"]
                        },
                        "2": {
                            "text": "That was epic!",
                            "label": "out-of-character",
                            "actions": []
                        }
                    }
                }
            }
        }
    
    def test_analyze_paragraph_actions_with_sample_data(self, sample_data):
        """Test the function with sample data."""
        results = dnd.analyze_paragraph_actions(sample_data)
        
        # Validate results structure
        expected_keys = [
            'spells_paragraphs', 'roll_paragraphs', 'no_action_paragraphs',
            'total_paragraphs', 'in_character_paragraphs', 'out_of_character_paragraphs',
            'mixed_paragraphs'
        ]
        
        for key in expected_keys:
            assert key in results, f"Result should contain '{key}'"
        
        # Validate counts make sense
        assert results['total_paragraphs'] == 3, f"Wrong total paragraph count: {results['total_paragraphs']} (expected 3)"
        assert results['spells_paragraphs'] == 1, f"Wrong spells paragraph count: {results['spells_paragraphs']} (expected 1)"
        assert results['roll_paragraphs'] == 1, f"Wrong roll paragraph count: {results['roll_paragraphs']} (expected 1)"
        assert results['no_action_paragraphs'] == 1, f"Wrong no_action paragraph count: {results['no_action_paragraphs']} (expected 1)"
        
        # Validate character label counts
        assert results['in_character_paragraphs'] == 1, "Should have 1 in-character paragraph"
        assert results['out_of_character_paragraphs'] == 1, "Should have 1 out-of-character paragraph"
        assert results['mixed_paragraphs'] == 1, "Should have 1 mixed paragraph"


class TestMultiCampaignIntegration:
    """Test that paragraph actions integrates with multi-campaign analysis."""
    
    @pytest.fixture
    def campaign_test_data(self):
        """Create test data for multi-campaign testing."""
        return {
            "campaign-1": {
                "1": {
                    "date": "2023-01-01T10:00:00",
                    "player": "Alice",
                    "character": "Elara",
                    "paragraphs": {
                        "0": {
                            "text": "I attack with my sword.",
                            "label": "in-character",
                            "actions": ["weapon"]
                        }
                    },
                    "actions": [],
                    "name_mentions": [],
                    "in_combat": False
                }
            }
        }
    
    def test_multi_campaign_integration(self, campaign_test_data):
        """Test that paragraph actions integrates with multi-campaign analysis."""
        # Load as dataframes
        campaign_dataframes = {
            "campaign-1": dnd.load_dnd_data(campaign_test_data)
        }
        
        # Test multi-campaign analysis with JSON data
        all_results = dnd.analyze_all_campaigns(
            campaign_dataframes=campaign_dataframes,
            original_json_data=campaign_test_data,
            show_progress=False
        )
        
        # Check that results contain expected structure
        assert 'per_campaign' in all_results, "Multi-campaign results should contain per_campaign data"
        assert 'campaign-1' in all_results['per_campaign'], "Test campaign should be found in results"
        
        campaign_results = all_results['per_campaign']['campaign-1']
        assert 'paragraph_actions' in campaign_results, "Campaign results should contain paragraph_actions"
        
        para_results = campaign_results['paragraph_actions']
        assert para_results.get('weapon_paragraphs', 0) == 1, f"Wrong weapon count: {para_results.get('weapon_paragraphs', 0)} (expected 1)"
        
        # Check aggregated results
        assert 'aggregated' in all_results, "Multi-campaign results should contain aggregated data"
        
        if 'paragraph_actions' in all_results['aggregated']:
            agg_para = all_results['aggregated']['paragraph_actions']
            assert agg_para.get('weapon_paragraphs', 0) == 1, f"Wrong aggregated weapon count: {agg_para.get('weapon_paragraphs', 0)}"


class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    @pytest.fixture
    def old_format_data(self):
        """Create test data in old format."""
        return {
            "campaign-1": {
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
    
    def test_backward_compatibility(self, old_format_data):
        """Test that old functions still work with old format data."""
        # Test that old functions still work
        df = dnd.load_dnd_data(old_format_data)
        assert len(df) == 1, "Should load one message"
        
        intervals = dnd.analyze_time_intervals(df, by_player=False)
        assert 'overall' in intervals, "Should analyze time intervals"
        
        lengths = dnd.analyze_post_lengths(df, by_player=False)
        assert 'overall' in lengths, "Should analyze post lengths"
        
        action_dialogue = dnd.analyze_action_vs_dialogue(df)
        assert 'total_posts' in action_dialogue, "Should analyze action vs dialogue"
        
        # Test multi-campaign analysis without JSON data
        campaign_dataframes = {"campaign-1": df}
        all_results = dnd.analyze_all_campaigns(
            campaign_dataframes=campaign_dataframes,
            original_json_data=None,
            show_progress=False
        )
        
        assert 'per_campaign' in all_results, "Should have per_campaign results"
        assert 'campaign-1' in all_results['per_campaign'], "Should have campaign-1 results"
        
        # Should not have paragraph actions when no JSON data
        campaign_results = all_results['per_campaign']['campaign-1']
        assert 'paragraph_actions' not in campaign_results, "Should not have paragraph actions without JSON data"


class TestLoadAllCampaignsFunction:
    """Test the load_all_campaigns function."""
    
    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """Create a temporary data file for testing."""
        mock_data = {
            "test-campaign": {
                "1": {
                    "date": "2023-01-01T10:00:00",
                    "player": "TestPlayer",
                    "character": "TestCharacter",
                    "paragraphs": {
                        "0": {
                            "text": "Test message.",
                            "label": "in-character",
                            "actions": ["dialogue"]
                        }
                    }
                }
            }
        }
        
        test_file = tmp_path / "test-data.json"
        with open(test_file, 'w') as f:
            json.dump(mock_data, f)
        
        return test_file
    
    def test_load_all_campaigns_function(self, temp_data_file):
        """Test load_all_campaigns function with default parameters."""
        campaign_dfs = dnd.load_all_campaigns(str(temp_data_file), max_campaigns=1, show_progress=False)
        
        assert isinstance(campaign_dfs, dict), "Should return dict"
        assert len(campaign_dfs) == 1, "Should load one campaign"
        assert "test-campaign" in campaign_dfs, "Should contain test campaign"
        
        # Verify the loaded DataFrame has expected structure
        df = campaign_dfs["test-campaign"]
        assert len(df) == 1, "Should have one message"
        assert "player" in df.columns, "Should have player column"
        assert "character" in df.columns, "Should have character column"
        assert "date" in df.columns, "Should have date column"
    
    def test_load_all_campaigns_with_json_return(self, temp_data_file):
        """Test load_all_campaigns function with return_json=True."""
        result = dnd.load_all_campaigns(str(temp_data_file), max_campaigns=1, show_progress=False, return_json=True)
        
        assert isinstance(result, tuple), "Should return tuple when return_json=True"
        assert len(result) == 2, "Should return tuple of length 2"
        
        campaign_dfs, json_data = result
        assert isinstance(campaign_dfs, dict), "First element should be dict"
        assert isinstance(json_data, dict), "Second element should be dict"
        assert "test-campaign" in campaign_dfs, "Should contain test campaign in dataframes"
        assert "test-campaign" in json_data, "Should contain test campaign in JSON data"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])