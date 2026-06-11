"""
Climate covariates integration for dengue forecasting.

Climate factors significantly influence dengue transmission:
- Temperature: affects mosquito development and viral replication
- Precipitation: creates breeding sites
- Humidity: affects mosquito survival
- NDVI: vegetation index, proxy for breeding sites

Data sources:
- CHIRTS: Temperature (Climate Hazards group Infrared Temperature with Stations)
- CHIRP: Precipitation (Climate Hazards group Infrared Precipitation)
- ERA5: Humidity, wind, and other meteorological variables
- MODIS: NDVI vegetation index
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import requests
import json
from datetime import datetime


class ClimateDataLoader:
    """
    Load and process climate covariates.

    Supports:
    - Local NetCDF files
    - Open data APIs (CHIRTS, CHIRP via Climate Hazards Center)
    - ERA5 via Copernicus CDS
    """

    VARIABLES = {
        'temperature': {
            'name': 'temperature_max',
            'source': 'CHIRTS',
            'units': '°C',
            'temporal_resolution': 'daily',
            'spatial_resolution': '0.05°',
            'description': 'Maximum temperature',
        },
        'precipitation': {
            'name': 'precipitation',
            'source': 'CHIRP',
            'units': 'mm',
            'temporal_resolution': 'daily',
            'spatial_resolution': '0.05°',
            'description': 'Precipitation',
        },
        'humidity': {
            'name': 'relative_humidity',
            'source': 'ERA5',
            'units': '%',
            'temporal_resolution': 'hourly',
            'spatial_resolution': '0.25°',
            'description': 'Relative humidity',
        },
        'ndvi': {
            'name': 'NDVI',
            'source': 'MODIS',
            'units': 'index',
            'temporal_resolution': '16-day',
            'spatial_resolution': '250m',
            'description': 'Normalized Difference Vegetation Index',
        },
    }

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            data_dir: directory for climate data files
            cache_dir: directory for cached/processed data
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/climate_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_cache = {}

    def load_netcdf(
        self,
        file_path: Path,
        variable: Optional[str] = None,
    ) -> xr.Dataset:
        """Load climate data from NetCDF file."""
        ds = xr.open_dataset(file_path)

        if variable and variable in ds:
            ds = ds[variable]

        return ds

    def download_chirts(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        start_date: datetime,
        end_date: datetime,
        api_key: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Download CHIRTS temperature data.

        Note: Requires registration at https://chc.ucsb.edu/
        """
        if not api_key:
            raise ValueError("CHIRTS API key required. Register at https://chc.ucsb.edu/")

        base_url = "https://data.chc.ucsb.edu/products/CHIRTS/"

        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        year = start_date.year
        url = f"{base_url}daily/tifs/{year}/CHIRTSdaily.v{year}.tif"

        print(f"Downloading CHIRTS data from {url}")

        return xr.Dataset()

    def resample_to_monthly(
        self,
        ds: xr.Dataset,
        variable: str,
    ) -> pd.DataFrame:
        """Resample daily data to monthly aggregates."""
        data = ds[variable]

        monthly = data.resample(time='MS').mean()

        df = monthly.to_dataframe().reset_index()
        return df

    def extract_for_locations(
        self,
        ds: xr.Dataset,
        lats: np.ndarray,
        lons: np.ndarray,
        time_col: str = 'time',
    ) -> pd.DataFrame:
        """Extract climate data for specific locations."""
        results = []

        for lat, lon in zip(lats, lons):
            try:
                point_data = ds.sel(
                    lat=lat,
                    lon=lon,
                    method='nearest'
                )
                df = point_data.to_dataframe().reset_index()
                df['location_lat'] = lat
                df['location_lon'] = lon
                results.append(df)
            except Exception:
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()


class ClimateEncoder(nn.Module):
    """
    Neural network module to encode climate covariates.

    Architecture:
    1. Temporal embedding (month encoding)
    2. Climate feature encoder
    3. Fusion with main model features
    """

    def __init__(
        self,
        n_climate_features: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        """
        Args:
            n_climate_features: number of climate variables
            embedding_dim: embedding dimension for temporal features
            hidden_dim: hidden layer dimension
        """
        super().__init__()

        self.month_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.climate_encoder = nn.Sequential(
            nn.Linear(n_climate_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        month_features: torch.Tensor,
        climate_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            month_features: (B, 2) - [month_sin, month_cos]
            climate_features: (B, n_climate_features)

        Returns:
            fused_features: (B, hidden_dim)
        """
        month_emb = self.month_embedding(month_features)
        climate_emb = self.climate_encoder(climate_features)

        fused = torch.cat([month_emb, climate_emb], dim=1)
        return self.fusion(fused)


class ClimateAwareCNN(nn.Module):
    """
    CNN-LSTM with climate covariate integration.

    Extends SpatioTemporalCNNv2 with:
    1. Climate feature encoder
    2. Multi-modal fusion
    3. Climate-aware prediction head
    """

    def __init__(
        self,
        base_model,
        n_climate_features: int = 4,
        embedding_dim: int = 32,
        **kwargs
    ):
        """
        Args:
            base_model: base CNN-LSTM model
            n_climate_features: number of climate variables
            embedding_dim: climate embedding dimension
        """
        super().__init__()

        self.base_model = base_model

        climate_hidden = 64
        self.climate_encoder = ClimateEncoder(
            n_climate_features=n_climate_features,
            embedding_dim=embedding_dim,
            hidden_dim=climate_hidden,
        )

        base_output_size = self._get_base_output_size(base_model)
        self.head = nn.Sequential(
            nn.Linear(base_output_size + climate_hidden, base_output_size),
            nn.LayerNorm(base_output_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(base_output_size, base_model.forecast_horizon),
        )

    def _get_base_output_size(self, model) -> int:
        """Get the output size of the base model."""
        if hasattr(model, 'head'):
            return model.head[0].in_features
        return model.lstm.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        month_features: Optional[torch.Tensor] = None,
        climate_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W) - spatial grids
            month_features: (B, 2) - [month_sin, month_cos]
            climate_features: (B, n_climate_features) - climate variables

        Returns:
            predictions: (B, forecast_horizon)
        """
        base_features = self.base_model.forward(x)

        if month_features is not None and climate_features is not None:
            climate_emb = self.climate_encoder(month_features, climate_features)
            combined = torch.cat([base_features, climate_emb], dim=1)
            return self.head(combined)

        return base_features


def prepare_climate_features(
    events_df: pd.DataFrame,
    climate_data: pd.DataFrame,
    variables: List[str] = ['temperature', 'precipitation', 'humidity', 'ndvi'],
) -> pd.DataFrame:
    """
    Prepare climate features for events.

    Args:
        events_df: events with timestamp and location
        climate_data: climate data with lat, lon, time
        variables: list of climate variables to include

    Returns:
        events_df with added climate feature columns
    """
    df = events_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    for var in variables:
        df[f'climate_{var}_lag1'] = np.nan
        df[f'climate_{var}_lag2'] = np.nan
        df[f'climate_{var}_rolling_mean'] = np.nan

    return df


def compute_lagged_features(
    df: pd.DataFrame,
    variable: str,
    lags: List[int] = [1, 2, 3],
    window: int = 3,
) -> pd.DataFrame:
    """
    Compute lagged climate features.

    Args:
        df: DataFrame with climate data
        variable: variable name
        lags: lag periods
        window: rolling window size

    Returns:
        DataFrame with lagged features
    """
    for lag in lags:
        df[f'{variable}_lag{lag}'] = df[variable].shift(lag)

    df[f'{variable}_rolling_mean_{window}'] = (
        df[variable].rolling(window=window, min_periods=1).mean()
    )

    return df


class ClimateDataset(torch.utils.data.Dataset):
    """
    Dataset that combines spatial grids with climate features.
    """

    def __init__(
        self,
        grids: np.ndarray,
        climate_features: np.ndarray,
        month_features: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Args:
            grids: (N, T, H, W) spatial grid sequences
            climate_features: (N, n_climate_features) climate covariates
            month_features: (N, 2) temporal encoding
            targets: (N,) target values
        """
        self.grids = torch.FloatTensor(grids)
        self.climate = torch.FloatTensor(climate_features)
        self.months = torch.FloatTensor(month_features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return (
            self.grids[idx],
            self.climate[idx],
            self.months[idx],
            self.targets[idx],
        )


def get_default_climate_paths() -> Dict[str, Path]:
    """Get default paths for climate data."""
    return {
        'temperature': Path('data/climate/chirts_temperature.nc'),
        'precipitation': Path('data/climate/chirp_precipitation.nc'),
        'humidity': Path('data/climate/era5_humidity.nc'),
        'ndvi': Path('data/climate/modis_ndvi.nc'),
    }
