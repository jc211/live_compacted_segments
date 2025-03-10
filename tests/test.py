import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from live_compacted_segments import TimeSeriesSamplerBuilder

wp.init()


# Create example time series data
def create_example_data():
    # Signal 1: 0 to 5 seconds
    times1 = np.linspace(0, 5, 500)
    values1 = np.stack(
        [
            np.sin(2 * np.pi * 1.0 * times1),
            np.cos(2 * np.pi * 1.0 * times1),
        ],
        axis=-1,
    )
    values1 = np.stack([values1, values1 * 0.1], axis=1)

    # Signal 2: 0 to 8 seconds with different frequency
    times2 = np.linspace(0, 8, 800)
    values2 = np.stack(
        [
            0.8 * np.sin(2 * np.pi * 2.0 * times2 + np.pi / 4),
            0.8 * np.cos(2 * np.pi * 2.0 * times2 + np.pi / 4),
        ],
        axis=-1,
    )
    values2 = np.stack([values2, values2 * 0.1], axis=1)

    # Signal 3: 0 to 10 seconds with different phase
    times3 = np.linspace(0, 10, 1000)
    values3 = np.stack(
        [
            1.2 * np.sin(2 * np.pi * 0.5 * times3 + np.pi / 2),
            1.2 * np.cos(2 * np.pi * 0.5 * times3 + np.pi / 2),
        ],
        axis=-1,
    )
    values3 = np.stack([values3, values3 * 0.1], axis=1)

    return [(times1, values1), (times2, values2), (times3, values3)]


def plot_sampler_results(sampler, results, metadata):
    # Get the full series data
    batch_size = sampler.batch_size
    dt = sampler.dt
    num_samples = sampler.num_samples
    full_data = sampler.get_full_series()

    # Determine the number of dimensions and values
    num_dims = sampler.dim_values
    num_values = sampler.num_values

    # Create a single figure with a grid of subplots
    fig, axes = plt.subplots(
        num_values, num_dims, figsize=(14 * num_dims, 8 * num_values), squeeze=False
    )

    # Plot each value and dimension in its corresponding subplot
    names = sampler.get_names(metadata)
    for val_id in range(num_values):
        for dim_id in range(num_dims):
            ax = axes[val_id, dim_id]
            offsets = []
            last_t = 0

            # Plot the concatenated signal with different colors for each original signal
            for i, (start, end) in enumerate(full_data["boundaries"]):
                offsets.append(last_t)
                ax.plot(
                    full_data["times"][start:end] + last_t,
                    full_data["values"][start:end][:, val_id, dim_id],
                )
                last_t = (
                    last_t + full_data["times"][end - 1] - full_data["times"][start]
                )

            # Plot the subsampled signals as points
            time_axis = np.arange(0, num_samples * dt, dt)
            time_segments = metadata["time_segments"].cpu().numpy()
            signal_index = metadata["signal_index"].cpu().numpy()
            colors = plt.cm.viridis(np.linspace(0, 1, batch_size))

            for batch_idx in range(batch_size):
                # Calculate the time positions for the subsampled points
                t_start = time_segments[batch_idx, 0]
                t_points = t_start + time_axis

                # Get the signal index for this subsampled segment
                signal_idx = signal_index[batch_idx]
                signal_name = names[batch_idx]

                ax.scatter(
                    t_points + offsets[signal_idx],
                    results[batch_idx, :][:, val_id, dim_id],
                    color=colors[batch_idx],
                    label=f"Batch {batch_idx} (Signal {signal_name}, Start={t_start:.2f})",
                    s=10,
                )

            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(f"Value {val_id}, Dimension {dim_id}")
            ax.legend()
            ax.grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


# Usage example
if __name__ == "__main__":
    # Create example data
    signals = create_example_data()

    # Build the sampler
    batch_size = 3
    num_samples = 100
    dt = 0.1
    sampler = (
        TimeSeriesSamplerBuilder()
        .add_series(signals[0][0], signals[0][1], "Sine 1Hz")
        .add_series(signals[1][0], signals[1][1], "Sine 2Hz")
        .add_series(signals[2][0], signals[2][1], "Sine 0.5Hz")
        .finalize(batch_size, num_samples, dt)
    )

    # Sample from the time series
    results, metadata = sampler.sample()
    names = sampler.get_names(metadata)

    # Display information about the series
    print("Series info:", sampler.get_series_info())

    # Display sampling metadata
    print("\nSampling metadata:")
    for batch_idx in range(batch_size):
        print(
            f"Batch {batch_idx}: Signal={names[batch_idx]}, "
            f"Time Segments={metadata['time_segments'][batch_idx, 0]:.2f}, "
            f"Index Bounds={metadata['idx_bounds'][batch_idx]}"
        )

    # Plot the results
    plot_sampler_results(sampler, results.cpu().numpy(), metadata)
