import numpy as np
import matplotlib.pyplot as plt
from live_compacted_segments.timeseries_sampler import TimeSeriesSampler
from live_compacted_segments.kernels import InterpolationMethod

def create_test_data():
    # Create test time series data
    times1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    values1 = np.array([
        [[1, 2], [3, 4]],  # First value
        [[5, 6], [7, 8]],  # Second value
        [[9, 10], [11, 12]],  # Third value
        [[13, 14], [15, 16]],  # Fourth value
    ], dtype=np.int32)

    times2 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    values2 = np.array([
        [[17, 18], [19, 20]],  # First value
        [[21, 22], [23, 24]],  # Second value
        [[25, 26], [27, 28]],  # Third value
    ], dtype=np.int32)

    return [(times1, values1), (times2, values2)]

def test_integer_sampler_zero_order_hold():
    # Create test data
    series_list = create_test_data()
    series_names = ["series1", "series2"]
    
    # Create sampler
    sampler = TimeSeriesSampler(
        batch_size=2,
        num_samples=4,
        dt=0.5,
        interpolation_method=InterpolationMethod.ZERO_ORDER_HOLD,
        series_list=series_list,
        series_names=series_names,
        output_type="int",
        device="cuda"
    )

    # Sample data
    output, metadata = sampler.sample()
    
    # Check output shape
    assert output.shape == (2, 4, 2, 2)  # (batch_size, num_samples, num_values, dim_values)
    
    # Check that all values are integers
    assert np.all(np.mod(output, 1) == 0)
    
    # Check that values are within the original data range
    assert np.min(output) >= np.min([np.min(v) for _, v in series_list])
    assert np.max(output) <= np.max([np.max(v) for _, v in series_list])

def test_integer_sampler_first_order_hold():
    # Create test data
    series_list = create_test_data()
    series_names = ["series1", "series2"]
    
    # Create sampler
    sampler = TimeSeriesSampler(
        batch_size=2,
        num_samples=4,
        dt=0.5,
        interpolation_method=InterpolationMethod.FIRST_ORDER_HOLD,
        series_list=series_list,
        series_names=series_names,
        output_type="int",
        device="cuda"
    )

    # Sample data
    output, metadata = sampler.sample()
    
    # Check output shape
    assert output.shape == (2, 4, 2, 2)
    
    # Check that all values are integers
    assert np.all(np.mod(output, 1) == 0)
    
    # Check that values are within the original data range
    assert np.min(output) >= np.min([np.min(v) for _, v in series_list])
    assert np.max(output) <= np.max([np.max(v) for _, v in series_list])

def test_integer_sampler_boundaries():
    # Create test data with specific boundary conditions
    times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    values = np.array([
        [[1, 1], [1, 1]],  # First value
        [[2, 2], [2, 2]],  # Second value
        [[3, 3], [3, 3]],  # Third value
    ], dtype=np.int32)
    
    series_list = [(times, values)]
    series_names = ["test_series"]
    
    # Create sampler
    sampler = TimeSeriesSampler(
        batch_size=1,
        num_samples=3,
        dt=0.5,
        interpolation_method=InterpolationMethod.ZERO_ORDER_HOLD,
        series_list=series_list,
        series_names=series_names,
        output_type="int",
        device="cuda"
    )

    # Sample data
    output, metadata = sampler.sample()
    
    # Check that values at boundaries match the original data
    assert np.array_equal(output[0, 0], values[0])  # First sample should match first value
    assert np.array_equal(output[0, -1], values[-1])  # Last sample should match last value

def test_integer_sampler_multiple_batches():
    # Create test data
    series_list = create_test_data()
    series_names = ["series1", "series2"]
    
    # Create sampler with larger batch size
    sampler = TimeSeriesSampler(
        batch_size=4,
        num_samples=3,
        dt=0.5,
        interpolation_method=InterpolationMethod.ZERO_ORDER_HOLD,
        series_list=series_list,
        series_names=series_names,
        output_type="int",
        device="cuda"
    )

    # Sample data multiple times
    outputs = []
    for _ in range(3):
        output, metadata = sampler.sample()
        outputs.append(output)
    
    # Check that all outputs have correct shape
    for output in outputs:
        assert output.shape == (4, 3, 2, 2)
    
    # Check that values are consistent (same input data should produce same output)
    assert np.array_equal(outputs[0], outputs[1])
    assert np.array_equal(outputs[1], outputs[2])

def test_integer_sampler_visualization():
    # Create test data with multiple series
    series_list = create_test_data()
    series_names = ["series1", "series2"]
    
    # Create sampler
    sampler = TimeSeriesSampler(
        batch_size=4,  # Increased batch size to see multiple samples
        num_samples=8,
        dt=0.2,
        interpolation_method=InterpolationMethod.ZERO_ORDER_HOLD,
        series_list=series_list,
        series_names=series_names,
        output_type="int",
        device="cuda"
    )
    
    # Sample data multiple times
    output, metadata = sampler.sample()
    output_cpu = output.cpu().numpy()
    
    # Extract time points for sampled data
    # time_segments is a tensor of shape (batch_size, num_samples, 2)
    # generate the time points for the sampled data
    sampled_times = sampler.warp_arrays["sample_times"].numpy()

    
    # Create a figure with subplots for each dimension
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Integer Sampler Visualization - All Dimensions and Series')
    
    # Plot each dimension
    for i in range(2):  # For each value index
        for j in range(2):  # For each dimension
            ax = axes[i, j]
            
            # Plot original data points for each series
            for series_idx, (times, values) in enumerate(series_list):
                ax.plot(times, values[:, i, j], 
                       marker='o', linestyle='-', 
                       label=f'Original {series_names[series_idx]}')
            
            # Plot sampled data points for each batch
            for batch_idx in range(4):
                ax.plot(sampled_times[batch_idx], output_cpu[batch_idx, :, i, j], 
                       '--', marker='s', alpha=0.5,
                       label=f'Sampled Data {batch_idx}')
            
            # Customize plot
            ax.set_title(f'Dimension [{i}, {j}]')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
            
            # Set y-axis limits with some padding
            y_min = min(
                min(np.min(values[:, i, j]) for _, values in series_list),
                np.min(output_cpu[:, :, i, j])
            )
            y_max = max(
                max(np.max(values[:, i, j]) for _, values in series_list),
                np.max(output_cpu[:, :, i, j])
            )
            ax.set_ylim(y_min - 0.5, y_max + 0.5)
    
    plt.tight_layout()
    plt.show()  # Display the figure in a window
    
    # Basic assertions to ensure the test is meaningful
    assert output_cpu.shape == (4, 8, 2, 2)  # Check shape with new batch size
    assert np.all(np.mod(output_cpu, 1) == 0)  # Check integer values
    assert np.min(output_cpu) >= min(np.min(v) for _, v in series_list)  # Check value range
    assert np.max(output_cpu) <= max(np.max(v) for _, v in series_list)

if __name__ == "__main__":
    # Run the visualization test
    test_integer_sampler_visualization()
    print("Visualization displayed in a window")