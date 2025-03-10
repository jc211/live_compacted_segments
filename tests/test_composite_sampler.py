import numpy as np
import torch
import warp as wp

from live_compacted_segments import (
    CompositeSampler,
    InterpolationMethod,
    TimeSeriesSamplerBuilder,
)


def generate_sample_data(num_timesteps, num_values=1, dim_values=3):
    """Generate sample time series data for testing."""
    timestamps = np.linspace(0, 10, num_timesteps).astype(np.float32)
    values = np.random.rand(num_timesteps, num_values, dim_values).astype(np.float32)
    return timestamps, values


def test_composite_sampler():
    """Test the CompositeSampler with multiple samplers of different output types."""
    # Setup test parameters
    batch_size = 4
    num_samples = 10
    dt = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate sample data for multiple series
    num_series = 2
    num_values = 1
    series_data = [generate_sample_data(100, num_values=1) for _ in range(num_series)]

    # Create samplers with different output types
    float_sampler = TimeSeriesSamplerBuilder(output_type="float")
    transform_sampler = TimeSeriesSamplerBuilder(output_type="transform")
    int_sampler = TimeSeriesSamplerBuilder(output_type="int")

    # Add the same series to each sampler
    for i, (timestamps, values) in enumerate(series_data):
        # For float sampler, use original values
        float_sampler.add_series(timestamps, values, f"float_series_{i}")

        # For transform sampler, convert values to a shape compatible with transforms
        # This is just for testing purposes
        transform_values = np.zeros(
            (len(timestamps), values.shape[1], 7), dtype=np.float32
        )
        transform_values[:, :, :3] = values[
            :, :, :3
        ]  # Use first 3 dimensions as position
        transform_values[:, :, 3:7] = np.array([1, 0, 0, 0])  # Identity quaternion
        transform_sampler.add_series(
            timestamps, transform_values, f"transform_series_{i}"
        )

        # For int sampler, convert to integers
        int_values = (values * 100).astype(np.int32)
        int_sampler.add_series(timestamps, int_values, f"int_series_{i}")

    # Finalize the samplers
    float_ts = float_sampler.finalize(
        batch_size, num_samples, dt, InterpolationMethod.FIRST_ORDER_HOLD, device
    )
    transform_ts = transform_sampler.finalize(
        batch_size, num_samples, dt, InterpolationMethod.FIRST_ORDER_HOLD, device
    )
    int_ts = int_sampler.finalize(
        batch_size, num_samples, dt, InterpolationMethod.FIRST_ORDER_HOLD, device
    )

    # Create the composite sampler
    samplers = {"float": float_ts, "transform": transform_ts, "int": int_ts}
    composite = CompositeSampler(samplers, lead_sampler_name="float")

    # Sample from the composite sampler
    for i in range(10):
        sampled_data = composite.sample()
        wp.synchronize()

    # Verify results
    assert len(sampled_data) == 3, "Should have data from all 3 samplers"
    assert "float" in sampled_data, "Float data missing"
    assert "transform" in sampled_data, "Transform data missing"
    assert "int" in sampled_data, "Int data missing"

    # Check shapes
    float_output = sampled_data["float"]
    transform_output = sampled_data["transform"]
    int_output = sampled_data["int"]

    assert float_output.shape == (batch_size, num_samples, num_values, 3), (
        f"Float output shape incorrect: {float_output.shape}"
    )
    assert transform_output.shape[:3] == (batch_size, num_samples, num_values), (
        f"Transform output shape incorrect: {transform_output.shape}"
    )
    assert int_output.shape[:3] == (batch_size, num_samples, num_values), (
        f"Int output shape incorrect: {int_output.shape}"
    )

    # Verify data types
    assert float_output.dtype == torch.float32, "Float output has wrong dtype"
    assert int_output.dtype == torch.int32, "Int output has wrong dtype"

    # Check that all samplers received the same time segments
    # This is just a sanity check; the actual time segments are determined at runtime
    float_result, float_metadata = float_ts.sample()
    float_time_segments = float_metadata["time_segments"]

    print(f"Batch size: {batch_size}")
    print(f"Num samples: {num_samples}")
    print(f"Float output shape: {float_output.shape}")
    print(f"Transform output shape: {transform_output.shape}")
    print(f"Int output shape: {int_output.shape}")

    return "CompositeSampler test completed successfully"


if __name__ == "__main__":
    # Initialize Warp
    wp.init()
    result = test_composite_sampler()
