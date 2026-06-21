"""
Unit tests for model modules.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np
import pandas as pd
from models.cnn_lstm import SpatioTemporalCNN, create_sequences
from models.nest import NESTIntensity, NESTForecaster
from models.hawkes import MultiDimensionalHawkes, HawkesBaseline


class TestCNNLSTM:
    """Tests for CNN-LSTM model."""

    @pytest.fixture
    def model(self):
        return SpatioTemporalCNN(grid_size=20, forecast_horizon=1, output_activation='softplus')

    def test_forward_shape(self, model):
        """Test that forward pass returns correct shape."""
        batch_size = 4
        seq_len = 12
        grid_size = 20
        
        x = torch.randn(batch_size, seq_len, grid_size, grid_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)

    def test_positive_output(self, model):
        """Test that softplus ensures non-negative output."""
        x = torch.randn(4, 12, 20, 20)
        output = model(x)
        
        assert output.min() >= 0, "Softplus should ensure non-negative output"
        assert torch.isfinite(output).all(), "Output should be finite"

    def test_different_activations(self):
        """Test different output activations."""
        for activation in ['softplus', 'exp', 'linear']:
            model = SpatioTemporalCNN(
                grid_size=20, 
                forecast_horizon=1, 
                output_activation=activation
            )
            x = torch.randn(4, 12, 20, 20)
            output = model(x)
            assert output.shape == (4, 1)
            assert torch.isfinite(output).all()


class TestCreateSequences:
    """Tests for sequence creation."""

    def test_sequence_shape(self):
        """Test that sequences have correct shape."""
        grid = np.random.rand(16, 16, 50)  # H, W, T
        X, y = create_sequences(grid, seq_len=12, forecast_horizon=1)
        
        expected_seqs = 50 - 12 - 1 + 1  # 38
        assert X.shape == (expected_seqs, 12, 16, 16)
        assert y.shape == (expected_seqs,)

    def test_non_negative_sequences(self):
        """Test that sequences are non-negative."""
        grid = np.abs(np.random.randn(16, 16, 50))
        X, y = create_sequences(grid)
        
        assert X.min() >= 0
        assert y.min() >= 0

    def test_empty_grid(self):
        """Test handling of empty grid."""
        grid = np.zeros((16, 16, 5))
        X, y = create_sequences(grid, seq_len=12)
        
        assert X.shape[0] == 0


class TestNESTIntensity:
    """Tests for NEST intensity model."""

    @pytest.fixture
    def model(self):
        return NESTIntensity(gs=20, output_activation='softplus')

    def test_forward_shape(self, model):
        """Test forward pass shape."""
        batch_size = 4
        seq_len = 12
        grid_size = 20
        
        x = torch.randn(batch_size, seq_len, grid_size, grid_size)
        output = model(x)
        
        assert output.shape == (batch_size, grid_size, grid_size)

    def test_positive_output(self, model):
        """Test that NEST produces non-negative output."""
        x = torch.randn(4, 12, 20, 20)
        output = model(x)
        
        assert output.min() >= 0, "NEST should produce non-negative intensity"
        assert torch.isfinite(output).all()


class TestMultiDimensionalHawkes:
    """Tests for Hawkes process."""

    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        hawkes = MultiDimensionalHawkes(n_regions=4)
        result = hawkes.fit([], [], [], max_iter=1)
        
        assert result is hawkes

    def test_fit_basic(self):
        """Test basic fitting."""
        hawkes = MultiDimensionalHawkes(n_regions=4)
        
        times = np.array([0.1, 0.5, 0.8, 1.2, 1.5])
        regions = np.array([0, 1, 0, 2, 1])
        weights = np.ones(5)
        
        hawkes.fit(times, regions, weights, max_iter=10)
        
        assert hawkes.mu.shape == (4,)
        assert hawkes.alpha.shape == (4, 4)
        assert hawkes.mu.min() >= 0

    def test_predict_intensity(self):
        """Test intensity prediction."""
        hawkes = MultiDimensionalHawkes(n_regions=2)
        
        # Fit with simple data
        times = np.array([0.1, 0.5, 0.8])
        regions = np.array([0, 1, 0])
        weights = np.ones(3)
        hawkes.fit(times, regions, weights, max_iter=10)
        
        # Predict intensity
        intensity = hawkes.predict_intensity(
            t=1.0,
            region_idx=0,
            past_times=times,
            past_regions=regions,
            past_weights=weights
        )
        
        assert intensity >= 0


class TestHawkesBaseline:
    """Tests for Hawkes baseline."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        hawkes = HawkesBaseline()
        
        # Create dummy events
        events = pd.DataFrame({
            'country': ['Vietnam', 'Vietnam', 'Thailand'],
            'timestamp': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01']),
            'case_count': [100, 150, 80]
        })
        
        hawkes.fit(events)
        
        assert hasattr(hawkes, 'is_fitted_')
        assert hawkes.is_fitted_ is True

    def test_predict_with_history(self):
        """Test prediction with historical counts."""
        hawkes = HawkesBaseline()
        
        events = pd.DataFrame({
            'country': ['Vietnam', 'Vietnam'],
            'timestamp': pd.to_datetime(['2020-01-01', '2020-02-01']),
            'case_count': [100, 150]
        })
        
        hawkes.fit(events)
        
        preds = hawkes.predict({'Vietnam': 150})
        assert 'Vietnam' in preds
        assert preds['Vietnam'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
