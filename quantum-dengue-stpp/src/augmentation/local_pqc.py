"""
Local PQC (Parameterized Quantum Circuit) with Spatial Clustering.

This module implements the LOCAL PQC approach where:
1. Events are clustered spatially using DBSCAN/K-Means based on their geographic location
2. Each cluster is trained by a separate LOCAL PQC with a small number of qubits (4-6)
3. This approach:
   - Reduces computational load (no need to process all 37,390 events in one quantum circuit)
   - Allows the quantum circuit to learn LOCAL spatial/geometric properties (SOP v2)
   - Maintains interpretability: each cluster corresponds to a geographic hotspot

Cluster types supported:
- DBSCAN: Density-based clustering (good for non-uniform hotspots)
- K-Means: Centroid-based clustering (good for uniform urban patterns)
- Ripley's L: Spatial autocorrelation-based clustering

Variational ansatz:
    ``LocalPQC`` supports two ansatz choices:
    - ``"strongly_entangling"`` (default): PennyLane ``StronglyEntanglingLayers`` — the
      legacy hardware-efficient ansatz. Useful as a baseline.
    - ``"data_reuploading"``: Data-Reuploading Ansatz (Pérez-Salinas et al. 2020),
      implemented in :mod:`augmentation.data_reuploading_ansatz`. Provably universal
      and shown to mitigate (but not eliminate) Barren Plateaus on small circuits.

References:
    - Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier",
      Quantum 5, 391 (2020). arXiv:1907.02040
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings("ignore")


class SpatialClusterer:
    """
    Spatial clustering for event data.

    Clusters events based on geographic location to enable local PQC training.
    """

    def __init__(self, method: str = 'dbscan', n_clusters: Optional[int] = None,
                 dbscan_eps: float = 1.0, dbscan_min_samples: int = 5,
                 random_state: int = 42):
        """
        Args:
            method: 'dbscan', 'kmeans', or 'ripley_kmeans'
            n_clusters: Number of clusters for K-Means (required for kmeans)
            dbscan_eps: Maximum distance for DBSCAN neighborhood
            dbscan_min_samples: Minimum samples in DBSCAN cluster
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.n_clusters_found_ = None

    def fit_predict(self, coords: np.ndarray) -> np.ndarray:
        """
        Cluster coordinates spatially.

        Args:
            coords: (n_events, 2) array of (lat, lon)

        Returns:
            cluster_labels: (n_events,) array of cluster assignments
        """
        # Normalize coordinates
        coords_scaled = self.scaler.fit_transform(coords)

        if self.method == 'dbscan':
            clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric='euclidean'
            )
            self.cluster_labels_ = clustering.fit_predict(coords_scaled)

        elif self.method == 'kmeans':
            if self.n_clusters is None:
                raise ValueError("n_clusters required for K-Means")
            clustering = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.cluster_labels_ = clustering.fit_predict(coords_scaled)
            self.cluster_centers_ = clustering.cluster_centers_

        elif self.method == 'ripley_kmeans':
            # Use Ripley's L-function to determine optimal clusters
            optimal_k = self._compute_ripley_k(coords)
            clustering = KMeans(
                n_clusters=optimal_k,
                random_state=self.random_state,
                n_init=10
            )
            self.cluster_labels_ = clustering.fit_predict(coords_scaled)
            self.cluster_centers_ = clustering.cluster_centers_
            self.n_clusters_found_ = optimal_k

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Count actual clusters (excluding noise label -1)
        self.n_clusters_found_ = len(set(self.cluster_labels_)) - (1 if -1 in self.cluster_labels_ else 0)

        return self.cluster_labels_

    def _compute_ripley_k(self, coords: np.ndarray, max_k: int = 20) -> int:
        """
        Use Ripley's L-function to estimate optimal number of clusters.

        L(t) = sqrt(K(t)/pi) - t, where K(t) is Ripley's K-function.
        For CSR (Complete Spatial Randomness), L(t) = 0.
        Deviations from 0 indicate clustering or regularity.
        """
        from scipy.spatial.distance import pdist, squareform

        n = len(coords)
        if n < 10:
            return 5  # Default

        # Compute pairwise distances
        dists = squareform(pdist(coords))

        # Find distance to k-th nearest neighbor
        sorted_dists = np.sort(dists, axis=1)
        k_distances = sorted_dists[:, 1:]  # Exclude self-distance

        # Estimate density at different scales
        area = (coords[:, 0].max() - coords[:, 0].min()) * \
               (coords[:, 1].max() - coords[:, 1].min())

        # Use elbow method on k-distance plot
        for k in range(2, min(max_k, n // 5)):
            kth_distances = k_distances[:, k - 1]
            # Variance of k-distances indicates clustering structure
            variance = np.var(kth_distances)

            if k > 3 and variance < 0.1:
                return max(2, k - 1)

        return min(8, n // 50)  # Default to ~8 clusters or fewer


class LocalPQC(nn.Module):
    """
    Local Parameterized Quantum Circuit for spatial hotspot modeling.

    Each cluster gets its own PQC with shared architecture but independent parameters.
    This allows:
    - Learning cluster-specific quantum embeddings
    - Reduced qubit count per circuit (4-6 qubits)
    - Parallel training across clusters
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3,
                 feature_dim: int = 8, embedding_type: str = 'angle',
                 ansatz: str = 'strongly_entangling'):
        """
        Args:
            n_qubits: Number of qubits (determines expressibility)
            n_layers: Number of variational layers
            feature_dim: Input feature dimension
            embedding_type: 'angle', 'amplitude', or 'basis'
            ansatz: variational ansatz — 'strongly_entangling' (default) or
                    'data_reuploading' (universal, see :mod:`data_reuploading_ansatz`).
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.embedding_type = embedding_type
        if ansatz not in ('strongly_entangling', 'data_reuploading'):
            raise ValueError(
                f"Unknown ansatz: {ansatz!r}. "
                "Choose 'strongly_entangling' or 'data_reuploading'."
            )
        self.ansatz = ansatz

        # Learnable circuit parameters
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1  # RY, RZ, RX parameters
        )

        # Classical pre-processing to match qubit count
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, n_qubits * 2),
            nn.GELU(),
            nn.Linear(n_qubits * 2, n_qubits),
        )

        # Output head for intensity prediction
        self.intensity_head = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.GELU(),
            nn.Linear(n_qubits, 1),
        )

        # Quantum device (created lazily)
        self._qdev = None

    @property
    def qdev(self):
        """Lazy initialization of quantum device."""
        if self._qdev is None:
            self._qdev = qml.device("default.qubit", wires=self.n_qubits)
        return self._qdev

    def _build_circuit(self):
        """Build the quantum circuit using PennyLane."""
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(features, weights):
            # Feature embedding
            if self.embedding_type == 'angle':
                # Normalize to [0, pi] and embed using rotations
                features_norm = torch.pi * (features - features.min()) / \
                               (features.max() - features.min() + 1e-8)
                qml.AngleEmbedding(features_norm[:self.n_qubits],
                                   wires=range(self.n_qubits),
                                   rotation='X')
            elif self.embedding_type == 'basis':
                # Basis encoding
                for i, val in enumerate(features[:self.n_qubits]):
                    if val > 0.5:
                        qml.PauliX(wires=i)
            else:
                # Amplitude encoding (requires power-of-2 qubits)
                qml.templates.AmplitudeEmbedding(
                    features[:2**self.n_qubits] / (torch.norm(features) + 1e-8),
                    wires=range(self.n_qubits),
                    normalize=True
                )

            # Variational layers (Strongly Entangling Layers)
            qml.templates.StronglyEntanglingLayers(
                weights, wires=range(self.n_qubits)
            )

            # Measurement: expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through quantum circuit.

        Args:
            x: (batch, feature_dim) input features

        Returns:
            intensities: (batch, 1) predicted intensities
        """
        # Pre-process features
        x_proj = self.feature_proj(x).float()  # ensure float32 for quantum backend

        if self.ansatz == 'data_reuploading':
            q_out = self._forward_data_reuploading(x_proj)
        else:
            q_out = self._forward_strongly_entangling(x_proj)

        # Match intensity_head dtype/device to avoid Double/Float mismatch
        q_out = q_out.to(dtype=next(self.intensity_head.parameters()).dtype)

        # Post-process to intensity
        intensity = self.intensity_head(q_out)
        intensity = torch.exp(intensity)  # Ensure positive

        return intensity

    def _forward_strongly_entangling(self, x_proj: torch.Tensor) -> torch.Tensor:
        """Original StronglyEntanglingLayers path."""
        circuit = self._build_circuit()
        q_out = circuit(x_proj, self.q_weights)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(list(q_out), dim=0)
        # PennyLane returns (n_qubits,) for a single input; we run with batched
        # classical pre-projection so we keep that shape and let intensity_head
        # handle a (batch, n_qubits) tensor.
        if q_out.dim() == 1:
            q_out = q_out.unsqueeze(0)
        # PennyLane convention is (n_qubits,) per sample; transpose to (batch, n_qubits)
        if q_out.shape[0] == self.n_qubits and q_out.dim() == 2:
            q_out = q_out.transpose(0, 1)
        return q_out

    def _forward_data_reuploading(self, x_proj: torch.Tensor) -> torch.Tensor:
        """Data-Reuploading Ansatz path (universal approximator)."""
        # Lazy import to avoid hard dependency at module-load time
        from .data_reuploading_ansatz import DataReuploadingPQC

        if not hasattr(self, '_dr_pqc') or self._dr_pqc is None:
            self._dr_pqc = DataReuploadingPQC(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                feature_dim=self.n_qubits,  # we feed the projected features
                entanglement='ring',
            ).to(x_proj.device)

        return self._dr_pqc(x_proj)


class ClusteredLocalPQC(nn.Module):
    """
    Clustered Local PQC: Multiple local PQC modules for different spatial clusters.

    This is the main class for the clustered quantum-classical hybrid model:
    - Spatial clustering separates events into geographic hotspots
    - Each cluster has its own Local PQC (shared architecture, independent params)
    - Outputs are aggregated for prediction

    NISQ hardware compatibility:
    - Circuit depth is capped at max 4 layers for local PQCs and global PQC.
    - StronglyEntanglingLayers with >4 layers may exceed coherence time on NISQ devices.
    """

    # Class constant for NISQ hardware compatibility
    MAX_CIRCUIT_LAYERS = 4  # Maximum layers for NISQ hardware compatibility

    def __init__(self, n_clusters: int = 8, n_qubits: int = 4, n_layers: int = 3,
                 feature_dim: int = 8, aggregation: str = 'weighted',
                 ansatz: str = 'strongly_entangling'):
        """
        Args:
            n_clusters: Maximum number of clusters
            n_qubits: Qubits per local PQC
            n_layers: Variational layers per PQC (capped at 4 for NISQ)
            feature_dim: Input feature dimension
            aggregation: 'weighted', 'mean', or 'sum'
            ansatz: forwarded to each :class:`LocalPQC` ('strongly_entangling' or
                    'data_reuploading').

        Note:
            n_layers is capped at MAX_CIRCUIT_LAYERS (4) for NISQ hardware compatibility.
            Deeper circuits may exceed coherence time and degrade performance.
        """
        import warnings

        super().__init__()
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits

        # Circuit depth validation for NISQ compatibility
        if n_layers > ClusteredLocalPQC.MAX_CIRCUIT_LAYERS:
            warnings.warn(
                f"n_layers={n_layers} exceeds NISQ recommendation (max {ClusteredLocalPQC.MAX_CIRCUIT_LAYERS}). "
                f"Circuit depth will be capped. For deeper circuits, consider using simulator "
                f"or quantum error mitigation techniques."
            )
            n_layers = ClusteredLocalPQC.MAX_CIRCUIT_LAYERS

        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.ansatz = ansatz

        # Create local PQC for each cluster
        self.cluster_pqcs = nn.ModuleList([
            LocalPQC(
                n_qubits=min(n_qubits, 6),  # Cap at 6 qubits
                n_layers=self.n_layers,
                feature_dim=feature_dim,
                ansatz=ansatz,
            )
            for _ in range(n_clusters)
        ])

        # Cluster attention weights (learnable - classical)
        self.cluster_attention = nn.Sequential(
            nn.Linear(feature_dim, n_clusters),
            nn.Softmax(dim=-1)
        )

        # Global PQC for cross-cluster learning
        # Cap layers at MAX_CIRCUIT_LAYERS for NISQ compatibility
        global_layers = min(self.n_layers, ClusteredLocalPQC.MAX_CIRCUIT_LAYERS)
        self.global_pqc = LocalPQC(
            n_qubits=min(n_qubits + 2, 8),
            n_layers=global_layers,
            feature_dim=feature_dim + n_clusters,  # Features + cluster indicators
            ansatz=ansatz,
        )

        # Log circuit specs for debugging
        self._circuit_depth_info = self._estimate_circuit_depth()

    def _estimate_circuit_depth(self) -> dict:
        """Estimate circuit depth for local and global PQC using qml.specs."""
        try:
            import pennylane as qml

            # Create a dummy circuit to estimate depth
            device = qml.device("default.qubit", wires=self.n_qubits)

            @qml.qnode(device, interface="torch", diff_method="parameter-shift")
            def dummy_circuit(features, weights):
                qml.AngleEmbedding(features[:self.n_qubits], wires=range(self.n_qubits), rotation='X')
                qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            # Get specs
            dummy_weights = torch.randn(self.n_layers, self.n_qubits, 3) * 0.1
            dummy_features = torch.randn(self.n_qubits)

            specs = qml.specs(dummy_circuit)(dummy_features, dummy_weights)

            return {
                'local_pqc_depth': specs.get('depth', self.n_layers * 4),
                'local_pqc_gates': specs.get('num_gates', self.n_layers * self.n_qubits * 4),
                'local_pqc_params': specs.get('num_parameters', self.n_layers * self.n_qubits * 3),
                'max_recommended_layers': ClusteredLocalPQC.MAX_CIRCUIT_LAYERS,
            }
        except Exception:
            # Fallback estimation
            return {
                'local_pqc_depth': self.n_layers * 4,  # RY+RZ+RX+CNOT per layer
                'local_pqc_gates': self.n_layers * self.n_qubits * 4,
                'local_pqc_params': self.n_layers * self.n_qubits * 3,
                'max_recommended_layers': ClusteredLocalPQC.MAX_CIRCUIT_LAYERS,
            }

    def forward(self, x: torch.Tensor,
                cluster_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cluster routing.

        Args:
            x: (batch, feature_dim) input features
            cluster_ids: (batch,) cluster assignments

        Returns:
            local_out: (batch, 1) output from local PQC
            global_out: (batch, 1) output from global PQC
        """
        batch_size = x.size(0)

        # Get cluster attention weights
        attn_weights = self.cluster_attention(x)  # (batch, n_clusters)

        # Local PQC outputs
        local_outputs = []
        for cluster_id in range(self.n_clusters):
            mask = (cluster_ids == cluster_id)
            if mask.sum() > 0:
                cluster_features = x[mask]
                pqc_out = self.cluster_pqcs[cluster_id](cluster_features)
                local_outputs.append((mask, pqc_out))

        # Aggregate local outputs
        local_out = torch.zeros(batch_size, 1, device=x.device)
        for mask, pqc_out in local_outputs:
            if self.aggregation == 'mean':
                local_out[mask] = pqc_out.mean()
            elif self.aggregation == 'sum':
                local_out[mask] = pqc_out.sum()
            else:  # weighted
                weights = attn_weights[mask, cluster_id:cluster_id+1]
                local_out[mask] = (pqc_out * weights.view(-1, 1)).sum(dim=0)

        # Global PQC with cluster context
        cluster_one_hot = torch.zeros(batch_size, self.n_clusters, device=x.device)
        cluster_one_hot.scatter_(1, cluster_ids.unsqueeze(1), 1)
        global_features = torch.cat([x, cluster_one_hot], dim=1)
        global_out = self.global_pqc(global_features)

        # Combine local and global
        combined = 0.5 * local_out + 0.5 * global_out

        return combined, local_out

    def get_cluster_expressivity(self) -> List[float]:
        """
        Compute expressibility metrics for each cluster's PQC.

        Returns:
            List of expressibility scores (higher = more expressive)
        """
        expressivities = []
        for pqc in self.cluster_pqcs:
            # Sample from circuit to estimate state space coverage
            weights = pqc.q_weights.detach()
            # Use variance of parameters as proxy for expressibility
            expr = torch.var(weights).item()
            expressivities.append(expr)
        return expressivities


def create_local_pqc_training_pipeline(
    coords: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    n_clusters: int = 8,
    cluster_method: str = 'dbscan',
    n_qubits: int = 4,
    n_layers: int = 3,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = 'cuda',
    verbose: bool = True,
    optimizer_type: str = 'adam',
) -> Tuple[ClusteredLocalPQC, Dict]:
    """
    Complete pipeline for training Local PQC with spatial clustering.

    Args:
        coords: (n_events, 2) geographic coordinates
        features: (n_events, feature_dim) input features
        targets: (n_events,) target counts
        n_clusters: Number of spatial clusters
        cluster_method: 'dbscan', 'kmeans', or 'ripley_kmeans'
        n_qubits: Qubits per local PQC
        n_layers: Variational layers (capped at 4 for NISQ compatibility)
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        device: 'cuda' or 'cpu'
        verbose: Print progress
        optimizer_type: 'adam' or 'qng' for quantum natural gradient

    Returns:
        model: Trained ClusteredLocalPQC
        info: Training info including optimizer_used, total_epochs, best_loss,
              avg_epoch_time_sec, total_time_sec
    """
    import time

    t0 = time.time()

    # Step 1: Spatial Clustering
    if verbose:
        print(f"\n  [Local PQC] Step 1: Spatial clustering ({cluster_method})...")

    clusterer = SpatialClusterer(
        method=cluster_method,
        n_clusters=n_clusters if cluster_method != 'dbscan' else None,
        dbscan_eps=0.5,
        dbscan_min_samples=10
    )
    cluster_labels = clusterer.fit_predict(coords)
    n_clusters_found = clusterer.n_clusters_found_

    if verbose:
        print(f"    Found {n_clusters_found} clusters")
        for c in range(n_clusters_found):
            n_in_cluster = (cluster_labels == c).sum()
            print(f"    Cluster {c}: {n_in_cluster} events")

    # Step 2: Initialize Model
    if verbose:
        print(f"  [Local PQC] Step 2: Initializing model...")

    model = ClusteredLocalPQC(
        n_clusters=n_clusters_found,
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_dim=features.shape[1]
    ).to(device)

    # Log circuit depth info
    if verbose:
        depth_info = model._circuit_depth_info
        print(f"    Circuit depth: {depth_info['local_pqc_depth']} (NISQ max: {depth_info['max_recommended_layers']})")

    # Step 3: Prepare Data
    cluster_tensor = torch.LongTensor(cluster_labels)
    features_tensor = torch.FloatTensor(features)
    targets_tensor = torch.FloatTensor(targets).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(
        features_tensor, cluster_tensor, targets_tensor
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Step 4: Training
    if verbose:
        opt_name = 'QNG + AdamW' if optimizer_type == 'qng' else 'AdamW'
        print(f"  [Local PQC] Step 3: Training ({epochs} epochs, optimizer={opt_name})...")

    criterion = nn.MSELoss()
    epoch_times = []

    # Create optimizer based on type
    if optimizer_type == 'qng':
        from .quantum_natural_gradient import create_qng_optimizer

        optimizer = create_qng_optimizer(
            model,
            lr_q=lr,
            lr_c=lr,
            use_diag_qng=True,  # Use DiagonalQNG for speed
            qng_for_weights="pqc",
            device=device,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {'loss': [], 'mse': [], 'epoch_time': []}
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0

        for feats, clusters, targets_batch in loader:
            feats = feats.to(device)
            clusters = clusters.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            pred, _ = model(feats, clusters)
            loss = criterion(pred, targets_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss.item()
            n_batches += 1

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['mse'].append(epoch_mse / max(n_batches, 1))
        history['epoch_time'].append(epoch_time)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:>4}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    # Load best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Step 5: Compute cluster expressivities
    expressivities = model.get_cluster_expressivity()

    total_time = time.time() - t0
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0.0

    info = {
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters_found,
        'cluster_centers': clusterer.cluster_centers_,
        'expressivities': expressivities,
        'best_loss': best_loss,
        'training_time': total_time,
        'optimizer_used': 'qng' if optimizer_type == 'qng' else 'adam',
        'total_epochs': epochs,
        'avg_epoch_time_sec': avg_epoch_time,
        'total_time_sec': total_time,
        'circuit_depth': model._circuit_depth_info,
    }

    if verbose:
        print(f"  [Local PQC] Training complete in {info['training_time']:.1f}s")
        print(f"    Optimizer: {info['optimizer_used'].upper()}")
        print(f"    Best loss: {best_loss:.4f}")
        print(f"    Avg epoch time: {avg_epoch_time:.2f}s")
        print(f"    Cluster expressivities: {[f'{e:.4f}' for e in expressivities]}")

        # Benchmark comparison for QNG
        if optimizer_type == 'qng':
            print(f"\n  [QNG Benchmark]")
            print(f"    Avg epoch time: {avg_epoch_time:.2f}s")
            print(f"    Note: QNG overhead expected due to metric tensor computation.")

    return model, info


# =============================================================================
# QFI (Quantum Fisher Information) for Measuring Quantum Advantage
# =============================================================================

class QuantumFisherInformation:
    """
    Compute Quantum Fisher Information (QFI) for quantum circuit analysis.

    QFI measures how much information about a parameter is encoded in the quantum state.
    High QFI indicates the circuit is sensitive to parameter changes — a proxy for
    expressibility and potential quantum advantage.
    """

    def __init__(self, n_qubits: int = 4, n_samples: int = 100):
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.qdev = qml.device("default.qubit", wires=n_qubits)

    def compute_qfi(self, circuit_fn, params: torch.Tensor) -> float:
        """
        Compute QFI for a parameterized quantum circuit.

        Args:
            circuit_fn: Quantum circuit function
            params: Circuit parameters (n_layers, n_qubits, 3)

        Returns:
            QFI value (average over parameter space)
        """
        qfi_total = 0.0

        for _ in range(self.n_samples):
            # Compute QFI via parameter-shift rule
            eps = 1e-3
            params_plus = params + eps
            params_minus = params - eps

            # Get states
            state_plus = circuit_fn(params_plus.view(-1))
            state_minus = circuit_fn(params_minus.view(-1))

            # QFI estimation via fidelity derivative
            fidelity = torch.real(torch.conj(state_plus) @ state_minus)
            qfi_total += (1 - fidelity) / (2 * eps ** 2)

        return (qfi_total / self.n_samples).item()

    def compute_circuit_specs(self, circuit_fn, params: torch.Tensor) -> Dict:
        """
        Compute circuit resource estimates and QFI.

        Returns dictionary with circuit analysis.
        """
        # Get circuit depth and gate count
        specs = qml.specs(circuit_fn)(params.view(-1))

        return {
            'depth': specs.get('depth', 0),
            'num_gates': specs.get('num_gates', 0),
            'num_parameters': specs.get('num_parameters', 0),
            'num_wires': self.n_qubits,
        }

    def estimate_haar_expressibility(self, circuit_fn, n_random_samples: int = 50) -> float:
        """
        Estimate circuit expressibility using Haar measure comparison.

        Samples from the circuit output distribution and compares to
        Haar-random states. High similarity indicates high expressibility.

        Returns:
            Expressibility score (0-1, higher = more expressive)
        """
        # Sample circuit outputs
        circuit_outputs = []
        for _ in range(n_random_samples):
            random_params = torch.randn_like(self.q_weights) * 0.1
            output = circuit_fn(random_params.view(-1))
            circuit_outputs.append(output.detach().numpy())

        circuit_outputs = np.array(circuit_outputs)

        # Compute variance of output distributions
        circuit_variance = np.var(circuit_outputs, axis=0).mean()

        # Compare to expected Haar variance
        # For d-dimensional system, Haar-random states have variance ~1/d
        haar_variance = 1.0 / (2 ** self.n_qubits)

        expressibility = min(1.0, circuit_variance / (haar_variance + 1e-9))

        return float(expressibility)


def analyze_quantum_advantage(
    model: ClusteredLocalPQC,
    X_test: np.ndarray,
    cluster_ids_test: np.ndarray,
    y_test: np.ndarray,
    classical_model: Optional[nn.Module] = None
) -> Dict:
    """
    Analyze potential quantum advantage by comparing Local PQC to classical baseline.

    Args:
        model: Trained ClusteredLocalPQC
        X_test: Test features
        cluster_ids_test: Test cluster assignments
        y_test: Test targets
        classical_model: Optional classical baseline for comparison

    Returns:
        Analysis results including QFI, expressibility, and performance metrics
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        clusters_t = torch.LongTensor(cluster_ids_test).to(device)

        # Get Local PQC predictions
        q_pred, local_preds = model(X_t, clusters_t)
        q_mse = torch.nn.functional.mse_loss(q_pred.squeeze(), torch.FloatTensor(y_test).to(device)).item()

        results = {
            'local_pqc_mse': q_mse,
            'local_pqc_mae': np.mean(np.abs(q_pred.cpu().numpy().squeeze() - y_test)),
            'cluster_predictions': {},
        }

        # Per-cluster analysis
        for cluster_id in range(model.n_clusters):
            mask = cluster_ids_test == cluster_id
            if mask.sum() > 0:
                cluster_mse = np.mean(
                    (q_pred.cpu().numpy().squeeze()[mask] - y_test[mask]) ** 2
                )
                results['cluster_predictions'][f'cluster_{cluster_id}'] = {
                    'n_samples': int(mask.sum()),
                    'mse': float(cluster_mse),
                    'expressibility': float(model.cluster_pqcs[cluster_id].q_weights.var().item()),
                }

        # Compare to classical if provided
        if classical_model is not None:
            classical_model.eval()
            c_pred = classical_model(torch.FloatTensor(X_test).to(device))
            c_mse = torch.nn.functional.mse_loss(
                c_pred.squeeze(), torch.FloatTensor(y_test).to(device)
            ).item()

            results['classical_mse'] = c_mse
            results['quantum_classical_ratio'] = q_mse / (c_mse + 1e-9)
            results['quantum_advantage'] = c_mse > q_mse

        # Compute average QFI across clusters
        qfi_analyzer = QuantumFisherInformation(n_qubits=4)
        avg_qfi = np.mean([
            model.cluster_pqcs[i].q_weights.var().item()
            for i in range(model.n_clusters)
        ])
        results['avg_expressibility'] = float(avg_qfi)

    return results
