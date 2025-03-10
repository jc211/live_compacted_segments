from typing import Literal
import numpy as np
from live_compacted_segments.timeseries_sampler import (
    TimeSeriesSampler,
    InterpolationMethod,
)


class TimeSeriesSamplerBuilder:
    """Builder for TimeSeriesSampler."""

    def __init__(
        self,
        batch_size: int,
        num_samples: int,
        dt: float,
        interpolation_method=InterpolationMethod.FIRST_ORDER_HOLD,
        device: str = "cuda",
        output_type: Literal["float", "transform", "int"] = "float",
    ):
        """Initialize the builder."""
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dt = dt
        self.interpolation_method = interpolation_method
        self.device = device
        self.output_type: Literal["float", "transform", "int"] = output_type
        self.series_list = []
        self.series_names = []

    def add_series(
        self, timestamps: np.ndarray, values: np.ndarray, name: str = None
    ) -> "TimeSeriesSamplerBuilder":
        """
        Add a time series to the sampler.

        Args:
            timestamps: 1D array of timestamps (must be sorted)
            values: 1D array of values corresponding to timestamps
            name: Optional name for the series

        Returns:
            self, for method chaining
        """
        # Validate inputs
        if len(timestamps) != len(values):
            raise ValueError("Timestamps and values arrays must have the same length")

        if len(timestamps) == 0:
            raise ValueError("Cannot add empty time series")

        # Check if timestamps are sorted
        if not np.all(np.diff(timestamps) >= 0):
            raise ValueError("Timestamps must be in ascending order")

        assert values.ndim == 3, (
            f"Values should be of shape (T, B, V), values is now of shape {values.shape}"
        )

        # Convert to float32 for warp compatibility
        timestamps = np.array(timestamps, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        # Generate a default name if none provided
        if name is None:
            name = f"series_{len(self.series_list)}"

        self.series_list.append((timestamps, values))
        self.series_names.append(name)

        return self

    def finalize(
        self,
    ) -> TimeSeriesSampler:
        """
        Create a TimeSeriesSampler with the added series.

        Returns:
            TimeSeriesSampler instance
        """
        if not self.series_list:
            raise ValueError("Cannot create a sampler with no time series")

        return TimeSeriesSampler(
            self.batch_size,
            self.num_samples,
            self.dt,
            self.interpolation_method,
            self.series_list,
            self.series_names,
            self.output_type,
            self.device,
        )
