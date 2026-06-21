"""
Unit tests for data loader module.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from data.loader import (
    validate_no_data_leakage,
    create_adaptive_spatial_grid,
    temporal_split,
)


class TestDataLeakage:
    """Tests for data leakage prevention."""

    def test_no_leakage_valid_splits(self):
        """Test that valid temporal splits pass validation."""
        train_df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', '2020-12-31', freq='ME')
        })
        val_df = pd.DataFrame({
            'timestamp': pd.date_range('2021-01-01', '2021-06-30', freq='ME')
        })
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2021-07-01', '2021-12-31', freq='ME')
        })
        
        assert validate_no_data_leakage(train_df, val_df, test_df) is True

    def test_leakage_detected_val(self):
        """Test that validation leakage is detected."""
        train_df = pd.DataFrame({
            'timestamp': pd.date_range('2021-06-01', '2021-12-31', freq='ME')
        })
        val_df = pd.DataFrame({
            'timestamp': pd.date_range('2021-01-01', '2021-06-30', freq='ME')
        })
        
        with pytest.raises(AssertionError):
            validate_no_data_leakage(train_df, val_df)

    def test_leakage_detected_test(self):
        """Test that test leakage is detected."""
        train_df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', '2020-12-31', freq='ME')
        })
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2021-01-01', '2021-12-31', freq='ME')
        })
        
        assert validate_no_data_leakage(train_df, test_df=test_df) is True


class TestTemporalSplit:
    """Tests for temporal split function."""

    def test_split_proportions(self):
        """Test that split returns correct proportions."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': range(100)
        })
        
        train, val, test = temporal_split(df, 0.7, 0.15, 0.15)
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_temporal_ordering(self):
        """Test that splits maintain temporal ordering."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': range(100)
        })
        
        train, val, test = temporal_split(df)
        
        assert train['timestamp'].max() <= val['timestamp'].min()
        assert val['timestamp'].max() <= test['timestamp'].min()


class TestAdaptiveSpatialGrid:
    """Tests for adaptive spatial gridding."""

    def test_grid_shape(self):
        """Test that grid has correct shape."""
        events = pd.DataFrame({
            'lat': np.random.uniform(-6, 23, 100),
            'lon': np.random.uniform(95, 141, 100),
            'timestamp': pd.date_range('2020-01-01', periods=100),
            'case_count': np.random.randint(0, 100, 100)
        })
        
        grid, glats, glons, norm_coords, params = create_adaptive_spatial_grid(
            events, grid_size=16, normalize_coords=True
        )
        
        assert grid.shape == (16, 16, 100)
        assert norm_coords.shape == (100, 2)
        assert params['normalized'] is True

    def test_normalized_coords_range(self):
        """Test that normalized coordinates are in [0, 1]."""
        events = pd.DataFrame({
            'lat': [0, 10, 20],  # range 0-20
            'lon': [100, 110, 120],  # range 100-120
            'timestamp': pd.date_range('2020-01-01', periods=3),
            'case_count': [10, 20, 30]
        })
        
        grid, glats, glons, norm_coords, params = create_adaptive_spatial_grid(
            events, grid_size=16, normalize_coords=True
        )
        
        assert norm_coords[:, 0].min() == 0.0
        assert norm_coords[:, 0].max() == 1.0
        assert norm_coords[:, 1].min() == 0.0
        assert norm_coords[:, 1].max() == 1.0

    def test_non_negative_cases(self):
        """Test that grid values are non-negative."""
        events = pd.DataFrame({
            'lat': np.random.uniform(-6, 23, 50),
            'lon': np.random.uniform(95, 141, 50),
            'timestamp': pd.date_range('2020-01-01', periods=50),
            'case_count': np.random.randint(0, 100, 50)
        })
        
        grid, _, _, _, _ = create_adaptive_spatial_grid(events)
        
        assert grid.min() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
