import warp as wp


# Constants for interpolation methods
class InterpolationMethod:
    ZERO_ORDER_HOLD = wp.constant(0)
    FIRST_ORDER_HOLD = wp.constant(1)


@wp.func
def binary_search(
    arr: wp.array(dtype=wp.float32), value: float, start: int, end: int
) -> int:
    """Binary search implementation for Warp."""
    low = start
    high = end - 1

    # Use a fixed number of iterations to avoid potential infinite loops
    max_iterations = 32  # Sufficient for arrays up to 2^32 elements

    # if out of bounds, return the closest bound
    if value < arr[low]:
        return low
    if value > arr[high]:
        return high - 1

    for i in range(max_iterations):
        if low > high:
            break

        mid = low + (high - low) // 2
        if arr[mid] < value:
            low = mid + 1
        else:
            high = mid - 1

    return high


@wp.func
def interpolate(
    times: wp.array(dtype=wp.float32),
    values: wp.array3d(dtype=wp.float32),
    t: float,
    idx: int,
    end: int,
    value_idx: int,
    dim_idx: int,
    interpolation_method: int,
) -> float:
    """Interpolate value at time t using the specified method."""
    # Zero-order hold: Use the value at the nearest previous point
    if interpolation_method == InterpolationMethod.ZERO_ORDER_HOLD:
        return values[idx, value_idx, dim_idx]

    # First-order hold: Linearly interpolate between the nearest points
    elif interpolation_method == InterpolationMethod.FIRST_ORDER_HOLD:
        if idx < end - 1:
            t0, t1 = times[idx], times[idx + 1]
            v0, v1 = (
                values[idx, value_idx, dim_idx],
                values[idx + 1, value_idx, dim_idx],
            )

            # Avoid division by zero
            if wp.abs(t1 - t0) < 1e-6:
                return v0

            alpha = (t - t0) / (t1 - t0)
            return v0 + alpha * (v1 - v0)
        else:
            return values[end - 1, value_idx, dim_idx]

    # Default return value
    return 0.0


@wp.func
def interpolate_transforms(
    times: wp.array(dtype=wp.float32),
    values: wp.array2d(dtype=wp.transformf),
    t: float,
    idx: int,
    end: int,
    value_idx: int,
    interpolation_method: int,
) -> wp.transformf:
    """Interpolate value at time t using the specified method."""
    # Zero-order hold: Use the value at the nearest previous point
    if interpolation_method == InterpolationMethod.ZERO_ORDER_HOLD:
        return values[idx, value_idx]

    # First-order hold: Linearly interpolate between the nearest points
    elif interpolation_method == InterpolationMethod.FIRST_ORDER_HOLD:
        if idx < end - 1:
            t0, t1 = times[idx], times[idx + 1]
            v0, v1 = (
                values[idx, value_idx],
                values[idx + 1, value_idx],
            )

            # Avoid division by zero
            if wp.abs(t1 - t0) < 1e-6:
                return v0

            alpha = (t - t0) / (t1 - t0)
            q0, q1 = (
                wp.transform_get_rotation(v0),
                wp.transform_get_rotation(v1),
            )

            p0, p1 = (
                wp.transform_get_translation(v0),
                wp.transform_get_translation(v1),
            )

            q_interp = wp.quat_slerp(wp.normalize(q0), wp.normalize(q1), alpha)
            q_interp = wp.normalize(q_interp)
            p_interp = p0 + alpha * (p1 - p0)

            return wp.transformf(p_interp, q_interp)

        else:
            return values[end - 1, value_idx]

    # Default return value
    return wp.transformf(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity(float))


@wp.kernel
def sample_time_ranges_kernel(
    times: wp.array(dtype=wp.float32),
    boundaries: wp.array2d(dtype=wp.int32),
    num_samples: int,
    dt: float,
    seed: int,
    signal_index: wp.array(dtype=wp.int32),
    time_segments: wp.array(dtype=wp.vec2f),
):
    """Kernel to sample time ranges for each batch."""
    # Get thread index for batch
    batch_idx = wp.tid()

    # Initialize random state using the seed and thread index
    state = wp.rand_init(
        seed + batch_idx * 10
    )  # Use a prime number for better distribution

    # Randomly select a signal from the concatenated data
    num_signals = boundaries.shape[0]
    signal_idx = wp.randi(state, 0, num_signals)
    start, end = boundaries[signal_idx, 0], boundaries[signal_idx, 1]

    # Store the signal index for this subsampled segment
    signal_index[batch_idx] = signal_idx

    # Calculate the required duration for the subsampled segment
    segment_duration = float(num_samples - 1) * dt

    # Get time bounds for the selected signal
    t_min = times[start]
    t_max = times[end - 1]

    # Adjust the maximum start time to ensure the segment fits within the signal
    max_start_time = t_max - segment_duration

    # If the signal is shorter than the required duration, start at the beginning
    if max_start_time <= t_min:
        t_start = t_min
        t_end = t_min + segment_duration
        if t_end > t_max:
            t_end = t_max  # Clamp to signal end
    else:
        # Randomly select a start time that ensures the segment fits
        t_start = t_min + wp.randf(state) * (max_start_time - t_min)
        t_end = t_start + segment_duration

    # Store the start and end times
    time_segments[batch_idx] = wp.vec2f(t_start, t_end)


@wp.kernel
def find_indices_kernel(
    times: wp.array(dtype=wp.float32),
    boundaries: wp.array2d(dtype=wp.int32),
    signal_index: wp.array(dtype=wp.int32),
    time_segments: wp.array(dtype=wp.vec2f),
    idx_bounds: wp.array2d(dtype=wp.int32),
):
    """Kernel to find the index bounds for each batch."""
    # Get thread index for batch
    batch_idx = wp.tid()

    # Get the signal index and time range for this batch
    signal_idx = signal_index[batch_idx]
    t_start, t_end = time_segments[batch_idx][0], time_segments[batch_idx][1]

    # Get the boundaries for this signal
    start, end = boundaries[signal_idx, 0], boundaries[signal_idx, 1]

    # Find the indices for interpolation using binary search
    idx_start = binary_search(times, t_start, start, end)
    idx_end = binary_search(times, t_end, start, end) + 1

    # Ensure idx_start is valid
    if idx_start < start:
        idx_start = start

    # Store the indices
    idx_bounds[batch_idx, 0] = idx_start
    idx_bounds[batch_idx, 1] = idx_end


@wp.kernel
def calculate_indices_kernel(
    times: wp.array(dtype=wp.float32),
    dt: float,
    boundaries: wp.array2d(dtype=wp.int32),
    signal_index: wp.array(dtype=wp.int32),
    time_segments: wp.array(dtype=wp.vec2f),
    idx_bounds: wp.array2d(dtype=wp.int32),
    output_indices: wp.array2d(dtype=wp.int32),
):
    """Kernel to calculate indices for the sampling."""
    # Get thread indices for batch and sample
    batch_idx, sample_idx = wp.tid()

    # Get the metadata for this batch
    signal_idx = signal_index[batch_idx]
    t_start = time_segments[batch_idx][0]
    t = t_start + float(sample_idx) * dt

    # Get the boundaries for this signal
    start, end = boundaries[signal_idx, 0], boundaries[signal_idx, 1]

    # Get the index bounds we've precomputed
    idx_start = idx_bounds[batch_idx, 0]
    idx_end = idx_bounds[batch_idx, 1]

    # First check if our time is before the first index or after the last
    if t < times[idx_start]:
        # Before first index, clamp to first value
        output_indices[batch_idx, sample_idx] = idx_start
        return

    if t > times[idx_end]:
        # After last index, clamp to last value
        output_indices[batch_idx, sample_idx] = idx_end
        return

    # Otherwise, find the exact index using a binary search within our precomputed bounds
    idx = binary_search(times, t, idx_start, idx_end + 1)

    # Ensure idx is valid
    if idx < start:
        idx = start
    if idx >= end:
        idx = end - 1

    output_indices[batch_idx, sample_idx] = idx


@wp.kernel
def interpolate_values_kernel(
    times: wp.array(dtype=wp.float32),
    dt: float,
    values: wp.array3d(dtype=wp.float32),
    time_segments: wp.array(dtype=wp.vec2f),
    output_indices: wp.array2d(dtype=wp.int32),
    boundaries: wp.array2d(dtype=wp.int32),
    signal_index: wp.array(dtype=wp.int32),
    output: wp.array4d(dtype=wp.float32),
    interpolation_method: int,
):
    """Kernel to interpolate values using the calculated indices."""
    # Get thread indices for batch, sample, and value
    batch_idx, sample_idx, value_idx = wp.tid()

    dim_values = output.shape[3]
    signal_idx = signal_index[batch_idx]
    end = boundaries[signal_idx, 1]
    t_start = time_segments[batch_idx][0]
    t = t_start + float(sample_idx) * dt

    # Get the calculated index and clamping status
    idx = output_indices[batch_idx, sample_idx]
    # Perform interpolation for non-clamped values
    for i in range(dim_values):
        output[batch_idx, sample_idx, value_idx, i] = interpolate(
            times, values, t, idx, end, value_idx, i, interpolation_method
        )


@wp.kernel
def interpolate_transforms_kernel(
    times: wp.array(dtype=wp.float32),
    dt: float,
    values: wp.array2d(dtype=wp.transformf),
    time_segments: wp.array(dtype=wp.vec2f),
    output_indices: wp.array2d(dtype=wp.int32),
    boundaries: wp.array2d(dtype=wp.int32),
    signal_index: wp.array(dtype=wp.int32),
    output: wp.array3d(dtype=wp.transformf),
    interpolation_method: int,
):
    """Kernel to interpolate values using the calculated indices."""
    # Get thread indices for batch, sample, and value
    batch_idx, sample_idx, value_idx = wp.tid()

    signal_idx = signal_index[batch_idx]
    end = boundaries[signal_idx, 1]
    t_start = time_segments[batch_idx][0]
    t = t_start + float(sample_idx) * dt

    # Get the calculated index and clamping status
    idx = output_indices[batch_idx, sample_idx]
    # Perform interpolation for non-clamped values
    output[batch_idx, sample_idx, value_idx] = interpolate_transforms(
        times, values, t, idx, end, value_idx, interpolation_method
    )


@wp.kernel
def interpolate_int_kernel(
    times: wp.array(dtype=wp.float32),
    dt: float,
    values: wp.array3d(dtype=wp.int32),
    time_segments: wp.array(dtype=wp.vec2f),
    output_indices: wp.array2d(dtype=wp.int32),
    boundaries: wp.array2d(dtype=wp.int32),
    signal_index: wp.array(dtype=wp.int32),
    output: wp.array4d(dtype=wp.int32),
    interpolation_method: int,
):
    """Kernel to interpolate values using the calculated indices."""
    # Get thread indices for batch, sample, and value
    batch_idx, sample_idx, value_idx = wp.tid()

    dim_values = output.shape[3]
    signal_idx = signal_index[batch_idx]
    end = boundaries[signal_idx, 1]
    t_start = time_segments[batch_idx][0]
    t = t_start + float(sample_idx) * dt

    # Get the calculated index and clamping status
    idx = output_indices[batch_idx, sample_idx]
    # Perform interpolation for non-clamped values
    for i in range(dim_values):
        output[batch_idx, sample_idx, value_idx, i] = values[idx, value_idx, i]


@wp.kernel
def calculate_sample_times_kernel(
    time_segments: wp.array(dtype=wp.vec2f),
    dt: float,
    output_times: wp.array2d(dtype=wp.float32),
):
    """Kernel to calculate the time at each sample point."""
    # Get thread indices for batch and sample
    batch_idx, sample_idx = wp.tid()
    
    # Get the start time for this batch
    t_start = time_segments[batch_idx][0]
    
    # Calculate the time for this sample
    t = t_start + float(sample_idx) * dt
    
    # Store the time
    output_times[batch_idx, sample_idx] = t
