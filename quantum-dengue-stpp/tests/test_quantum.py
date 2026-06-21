"""
Unit tests for ZINB loss and Local PQC modules.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np
from models.zinb_loss import (
    ZeroInflatedNegativeBinomialLoss,
    HybridQuantumZINB,
    SpatialZINBGridLoss,
    compute_zinb_metrics
)
from augmentation.local_pqc import (
    SpatialClusterer,
    LocalPQC,
    ClusteredLocalPQC,
)


class TestZINBLoss:
    """Tests for ZINB loss."""

    @pytest.fixture
    def loss_fn(self):
        return ZeroInflatedNegativeBinomialLoss(learn_theta=True)

    def test_forward(self, loss_fn):
        """Test forward pass."""
        batch_size = 8
        n_cells = 100
        
        pred_mu = torch.randn(batch_size, n_cells) * 2 + 5
        pred_pi = torch.randn(batch_size, n_cells) * 0.3 - 0.5
        target = torch.randint(0, 50, (batch_size, n_cells)).float()
        
        loss = loss_fn(pred_mu, pred_pi, target)
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_metrics(self, loss_fn):
        """Test metrics computation."""
        pred_mu = torch.randn(8, 100) * 2 + 5
        pred_pi = torch.randn(8, 100) * 0.3 - 0.5
        target = torch.randint(0, 50, (8, 100)).float()
        
        metrics = compute_zinb_metrics(
            pred_mu, pred_pi, target,
            theta=loss_fn.theta.item()
        )
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'zero_accuracy' in metrics
        assert metrics['zero_accuracy'] >= 0


class TestSpatialZINBGridLoss:
    """Tests for spatial ZINB loss."""

    def test_forward(self):
        """Test forward pass."""
        loss_fn = SpatialZINBGridLoss(spatial_smooth_weight=0.1)
        
        pred_mu = torch.randn(4, 400) * 2 + 5
        pred_pi = torch.randn(4, 400) * 0.3 - 0.5
        target = torch.randint(0, 50, (4, 400)).float()
        
        loss = loss_fn(pred_mu, pred_pi, target, grid_size=20)
        
        # Check loss is a tensor that can be converted to float
        assert hasattr(loss, 'item'), "Loss should be a tensor"
        loss_val = loss.sum().item() if loss.numel() > 1 else loss.item()
        assert loss_val >= 0


class TestSpatialClusterer:
    """Tests for spatial clustering."""

    def test_kmeans_clustering(self):
        """Test K-Means clustering."""
        np.random.seed(42)
        n_samples = 200
        
        # Create 3 clusters
        coords = np.vstack([
            np.random.randn(70, 2) + [0, 0],
            np.random.randn(70, 2) + [5, 5],
            np.random.randn(60, 2) + [10, 0],
        ])
        
        clusterer = SpatialClusterer(method='kmeans', n_clusters=3)
        labels = clusterer.fit_predict(coords)
        
        assert len(np.unique(labels)) == 3

    def test_dbscan_clustering(self):
        """Test DBSCAN clustering."""
        np.random.seed(42)
        coords = np.random.randn(100, 2)
        
        clusterer = SpatialClusterer(method='dbscan', dbscan_eps=0.5, dbscan_min_samples=5)
        labels = clusterer.fit_predict(coords)
        
        assert labels is not None


class TestLocalPQC:
    """Tests for Local PQC."""

    def test_initialization(self):
        """Test PQC initialization."""
        pqc = LocalPQC(n_qubits=4, n_layers=2, feature_dim=8)
        
        assert pqc.n_qubits == 4
        assert pqc.n_layers == 2
        assert pqc.q_weights.shape == (2, 4, 3)

    def test_clustered_pqc_initialization(self):
        """Test Clustered Local PQC initialization."""
        model = ClusteredLocalPQC(
            n_clusters=3,
            n_qubits=4,
            n_layers=2,
            feature_dim=8
        )
        
        assert len(model.cluster_pqcs) == 3
        assert model.n_clusters == 3

    def test_cluster_expressivity(self):
        """Test cluster expressivity computation."""
        model = ClusteredLocalPQC(n_clusters=3, n_qubits=4, n_layers=2, feature_dim=8)
        
        expressivities = model.get_cluster_expressivity()
        
        assert len(expressivities) == 3
        assert all(isinstance(e, float) for e in expressivities)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
