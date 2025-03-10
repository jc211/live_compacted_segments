from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp
import torch
from live_compacted_segments.kernels import (
    find_indices_kernel,
    sample_time_ranges_kernel,
    calculate_indices_kernel,
    interpolate_values_kernel,
    interpolate_transforms_kernel,
    interpolate_int_kernel,
    InterpolationMethod,
    calculate_sample_times_kernel,
)


class TimeSeriesSampler:
    """
    A class for sampling time series data with GPU acceleration using Warp.
    TimeSeriesSampler should be created using TimeSeriesSamplerBuilder.
    """

    def __init__(
        self,
        batch_size: int,
        num_samples: int,
        dt: float,
        interpolation_method,
        series_list: list[tuple[np.ndarray, np.ndarray]],
        series_names: list[str],
        output_type: Literal["float", "transform", "int"],
        device: str,
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dt = dt
        self.output_type = output_type
        self.device = device
        self.series_names = series_names
        self.interpolation_method = interpolation_method
        self.num_series = len(series_list)
        assert self.num_series, "Empty Series"

        self.num_values, self.dim_values = series_list[0][1].shape[1:]

        # Concatenate the time series and store boundaries
        self._concatenate_series(series_list)

        # Warp arrays will be created when needed
        self._create_warp_arrays(
            batch_size, num_samples, self.num_values, self.dim_values, self.device
        )
        self.time_sampled_event = wp.Event()
        self.finished_sampling = wp.Event()

    def sample_time_ranges(self):
        batch_size = self.batch_size
        num_samples = self.num_samples
        dt = self.dt
        seed = np.random.randint(0, 100000)

        wp.launch(
            kernel=sample_time_ranges_kernel,
            dim=batch_size,
            inputs=[
                self.warp_arrays["times"],
                self.warp_arrays["boundaries"],
                num_samples,
                dt,
                seed,
                self.warp_arrays["signal_index"],
                self.warp_arrays["time_segments"],
            ],
        )
        wp.record_event(self.time_sampled_event)
        time_segments = self.warp_arrays["time_segments"]
        signal_index = self.warp_arrays["signal_index"]
        return time_segments, signal_index

    def sample_from_time_segments(
        self, time_segments: wp.array, signal_index: wp.array
    ):
        batch_size = self.batch_size
        num_samples = self.num_samples
        num_values = self.num_values

        # Calculate sample times
        wp.launch(
            kernel=calculate_sample_times_kernel,
            dim=(batch_size, num_samples),
            inputs=[
                time_segments,
                self.dt,
                self.warp_arrays["sample_times"],
            ],
        )

        wp.launch(
            kernel=find_indices_kernel,
            dim=batch_size,
            inputs=[
                self.warp_arrays["times"],
                self.warp_arrays["boundaries"],
                signal_index,
                time_segments,
                self.warp_arrays["idx_bounds"],
            ],
        )

        # Step 4: Sample values kernel
        wp.launch(
            kernel=calculate_indices_kernel,
            dim=(batch_size, num_samples),
            inputs=[
                self.warp_arrays["times"],
                self.dt,
                self.warp_arrays["boundaries"],
                signal_index,
                time_segments,
                self.warp_arrays["idx_bounds"],
                self.warp_arrays["output_indices"],
            ],
        )

        if self.output_type == "float":
            wp.launch(
                kernel=interpolate_values_kernel,
                dim=(batch_size, num_samples, num_values),
                inputs=[
                    self.warp_arrays["times"],
                    self.dt,
                    self.warp_arrays["values"],
                    time_segments,
                    self.warp_arrays["output_indices"],
                    self.warp_arrays["boundaries"],
                    signal_index,
                    self.warp_arrays["output"],
                    self.interpolation_method,
                ],
            )
        elif self.output_type == "transform":
            wp.launch(
                kernel=interpolate_transforms_kernel,
                dim=(batch_size, num_samples, num_values),
                inputs=[
                    self.warp_arrays["times"],
                    self.dt,
                    self.warp_arrays["values"],
                    time_segments,
                    self.warp_arrays["output_indices"],
                    self.warp_arrays["boundaries"],
                    signal_index,
                    self.warp_arrays["output"],
                    self.interpolation_method,
                ],
            )
        elif self.output_type == "int":
            wp.launch(
                kernel=interpolate_int_kernel,
                dim=(batch_size, num_samples, num_values),
                inputs=[
                    self.warp_arrays["times"],
                    self.dt,
                    self.warp_arrays["values"],
                    time_segments,
                    self.warp_arrays["output_indices"],
                    self.warp_arrays["boundaries"],
                    signal_index,
                    self.warp_arrays["output"],
                    self.interpolation_method,
                ],
            )
        else:
            raise RuntimeError(f"output_type {self.output_type} not supported")
        wp.record_event(self.finished_sampling)

    def sample(
        self,
    ) -> tuple[torch.Tensor, dict]:
        time_segments, signal_index = self.sample_time_ranges()
        self.sample_from_time_segments(time_segments, signal_index)

        metadata = {
            "signal_index": wp.to_torch(self.warp_arrays["signal_index"]),
            "time_segments": wp.to_torch(self.warp_arrays["time_segments"]),
            "idx_bounds": wp.to_torch(self.warp_arrays["idx_bounds"]),
        }
        output = wp.to_torch(self.warp_arrays["output"])
        return output, metadata

    def get_names(self, metadata):
        return [
            self.series_names[idx] for idx in metadata["signal_index"].cpu().numpy()
        ]

    def get_series_info(self) -> dict:
        """
        Get information about the time series.

        Returns:
            Dictionary containing information about the series
        """
        series_info = []
        for i in range(self.num_series):
            start, end = self.boundaries[i]
            series_info.append(
                {
                    "name": self.series_names[i],
                    "start_index": int(start),
                    "end_index": int(end),
                    "length": int(end - start),
                    "start_time": float(self.concat_times[start]),
                    "end_time": float(self.concat_times[end - 1]),
                    "duration": float(
                        self.concat_times[end - 1] - self.concat_times[start]
                    ),
                }
            )

        return {
            "num_series": self.num_series,
            "total_points": len(self.concat_times),
            "series": series_info,
        }

    def get_full_series(self) -> dict:
        """
        Get the full concatenated time series data.

        Returns:
            Dictionary containing the concatenated time series and metadata
        """
        return {
            "times": self.concat_times,
            "values": self.concat_values,
            "boundaries": self.boundaries,
            "series_names": self.series_names,
        }

    def _concatenate_series(
        self, series_list: list[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Concatenate multiple time series into a single array with boundaries.

        Args:
            series_list: List of (timestamps, values) tuples
        """
        # Calculate the total length
        total_length = sum(len(timestamps) for timestamps, _ in series_list)

        # Pre-allocate arrays
        self.concat_times = np.zeros(total_length, dtype=np.float32)
        self.concat_values = np.zeros(
            (total_length, self.num_values, self.dim_values), dtype=np.float32
        )
        self.boundaries = np.zeros((len(series_list), 2), dtype=np.int32)

        # Fill the arrays
        idx = 0
        for i, (timestamps, values) in enumerate(series_list):
            n = len(timestamps)
            self.concat_times[idx : idx + n] = timestamps
            self.concat_values[idx : idx + n] = values
            self.boundaries[i] = [idx, idx + n]
            idx += n

    def _create_warp_arrays(
        self,
        batch_size: int,
        num_samples: int,
        num_values: int,
        dim_values: int,
        device: str,
    ) -> None:
        """
        Create Warp arrays for the concatenated signals and sampling metadata.

        Args:
            config: Sampling configuration
        """
        if self.output_type == "float":
            output = wp.zeros(
                (batch_size, num_samples, num_values, dim_values), dtype=wp.float32
            )  # (B, S, V, D)
            values = wp.array(self.concat_values, dtype=wp.float32)
            # (B, V, D)
        elif self.output_type == "transform":
            output = wp.zeros(
                (batch_size, num_samples, num_values), dtype=wp.transformf
            )  # (B, S, V, 7)
            values = wp.array(self.concat_values, dtype=wp.transformf)
            # (B, V)
        elif self.output_type == "int":
            output = wp.zeros(
                (batch_size, num_samples, num_values, dim_values), dtype=wp.int32
            )  # (B, S, V, D)
            values = wp.array(self.concat_values, dtype=wp.int32)
            # (B, V, D)
        else:
            raise RuntimeError("Output type not implemented")

        with wp.ScopedDevice(device):
            self.warp_arrays = {
                "times": wp.array(self.concat_times, dtype=wp.float32),
                "values": values,
                "boundaries": wp.array(self.boundaries, dtype=wp.int32),
                # Output arrays
                "signal_index": wp.zeros(batch_size, dtype=wp.int32),  # (B)
                "time_segments": wp.zeros(batch_size, dtype=wp.vec2f),  # (B, 2)
                "idx_bounds": wp.zeros((batch_size, 2), dtype=wp.int32),  # (B, 2)
                "output_indices": wp.zeros((batch_size, num_samples), dtype=wp.int32),  # (B, S)
                "output": output,
                "sample_times": wp.zeros((batch_size, num_samples), dtype=wp.float32),  # (B, S)
            }
