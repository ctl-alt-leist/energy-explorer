from datetime import datetime, timedelta
from typing import Tuple

import matplotlib.pyplot as plt
from numpy import nan_to_num, nanmax, zeros
from pandas import read_pickle

from energy_explorer.es_explorer import query_capacity_series
from energy_explorer.objects import CapacitySeries
from energy_explorer.paths import ENERGY_STORAGE_CLEANED_PATH


def compute_pair_correlation(
    series_m: CapacitySeries, series_n: CapacitySeries, max_shift: int, t_sigma: timedelta, t_delta: timedelta
) -> Tuple[float, int]:
    """
    Compute the maximum correlation and corresponding shift for two CapacitySeries objects based on their accelerations.
    Correlate only the region around the peak acceleration of series_m with the full acceleration of series_n. The shift
    is constrained to a maximum distance of `max_shift`.

    Args:
        series_m: The reference CapacitySeries object.
        series_n: The CapacitySeries object to be correlated against.
        max_shift: Maximum allowable shift in terms of number of time steps.
        t_sigma: Time window for smoothing.
        t_delta: Sampling time delta.

    Returns:
        max_corr: Maximum correlation value found within the allowable shift.
        shift: Corresponding shift value (in time steps) that gives the maximum correlation.
    """
    # Extract acceleration data and time series from both CapacitySeries
    accel_m = series_m.acceleration
    accel_n = series_n.acceleration

    # Find the index of the maximum acceleration in series_m
    m_peak = accel_m.argmax()

    # Determine the window of interest around the peak (from -3σ to +3σ)
    t_width = int((3 * t_sigma).total_seconds() / t_delta.total_seconds())
    start_index = max(0, m_peak - t_width)
    end_index = min(len(accel_m), m_peak + t_width)

    # Extract the kernel around the peak
    kernel = accel_m[start_index:end_index]

    # Make sure the kernel is free of NaNs (which could have resulted from smoothing or other processes)
    kernel = nan_to_num(kernel)

    # Calculate the length of the kernel
    kernel_length = len(kernel)

    # Initialize variables to track the best correlation and corresponding shift
    max_corr = -float("inf")
    best_shift = 0

    # Iterate over possible shifts (from -max_shift to +max_shift)
    for shift in range(-max_shift, max_shift + 1):
        # Determine the start and end indices of the window within accel_n based on the current shift
        shifted_center = m_peak + shift

        # Ensure that the center of the kernel is within valid bounds for accel_n
        if shifted_center - t_width < 0 or shifted_center + t_width >= len(accel_n):
            continue  # Skip invalid shifts

        # Extract the corresponding segment from accel_n
        segment_start = shifted_center - t_width
        segment_end = shifted_center + t_width
        segment_accel_n = accel_n[segment_start:segment_end]

        # Ensure the segment is the same length as the kernel
        if len(segment_accel_n) == kernel_length:
            # Compute the correlation by convolving the kernel with the segment
            correlation_value = (kernel * segment_accel_n).sum()

            # Update max_corr and best_shift if this correlation is greater
            if correlation_value > max_corr:
                max_corr = correlation_value
                best_shift = shift

    return max_corr, best_shift


def plot_correlation_matrices(max_corr_matrix, shift_matrix):
    """Plot the correlation and shift matrices, and scatter plot of correlation vs. shift."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Max Correlation Matrix
    axs[0].imshow(max_corr_matrix, cmap="viridis", aspect="auto")
    axs[0].set_title("Max Correlation Matrix")
    axs[0].set_xlabel("Series Index")
    axs[0].set_ylabel("Series Index")

    # Plot Shift Matrix
    axs[1].imshow(shift_matrix, cmap="coolwarm", aspect="auto")
    axs[1].set_title("Shift Matrix [Years]")
    axs[1].set_xlabel("Series Index")
    axs[1].set_ylabel("Series Index")

    # Scatter plot of Correlation vs. Shift
    x_shifts = []
    y_correlations = []
    for i in range(len(max_corr_matrix)):
        for j in range(i + 1):
            x_shifts.append(shift_matrix[i, j])
            y_correlations.append(max_corr_matrix[i, j])

    axs[2].scatter(x_shifts, y_correlations, color="b", alpha=0.7)
    axs[2].set_title("Correlation vs. Shift")
    axs[2].set_xlabel("Shift [Years]")
    axs[2].set_ylabel("Correlation")
    axs[2].axvline(x=0, color="gray", linestyle="--", linewidth=1)
    axs[2].axhline(y=1.0, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()


def main():
    """Find correlations between capacity acceleration of different groups."""

    # Data selection
    FUEL_TYPES = "Solar", "Battery"
    GROUP_BY = "facility_zipcode"
    SECTOR = "Residential"
    EXCLUSIVE = False

    # Time sampling
    T_DELTA = timedelta(days=1)
    T_SIGMA = timedelta(days=60)
    START_DATE = datetime(2001, 1, 1)
    END_DATE = datetime(2025, 1, 1)

    # Load and filter data
    es_frame = read_pickle(ENERGY_STORAGE_CLEANED_PATH)
    es_frame = es_frame[(es_frame.approval_date >= START_DATE) & (es_frame.approval_date < END_DATE)]

    # Retrieve the full set of capacity series
    all_series = []
    top_groups = query_capacity_series(
        es_frame=es_frame,
        fuel_type=FUEL_TYPES,
        exclusive=EXCLUSIVE,
        sector=SECTOR,
        group_by=GROUP_BY,
    )

    # Prepare and normalize each capacity series
    for group_id in top_groups.facility_zipcode.unique():
        group_data = top_groups[top_groups.facility_zipcode == group_id].sort_values(by="approval_date")
        date = group_data.approval_date.to_numpy()
        capacity = group_data.nameplate_capacity.cumsum().to_numpy() / 1000  # Convert to MW

        # Create and smooth the capacity series
        series = CapacitySeries(time=date, capacity=capacity)
        series.smooth(start=START_DATE, end=END_DATE, delta=T_DELTA, sigma=T_SIGMA)

        # Normalize capacity by its maximum value
        series.capacity /= nanmax(series.capacity)
        all_series.append(series)

    # Create correlation matrices
    n_series = len(all_series)
    max_corr_matrix = zeros((n_series, n_series))
    shift_matrix = zeros((n_series, n_series))

    # Calculate the max shift in points for 3 sigma
    max_shift = int((3 * T_SIGMA).total_seconds() / T_DELTA.total_seconds())

    # Iterate over each pair of series to compute correlations
    for m in range(n_series):
        for n in range(m + 1):  # n <= m ensures only lower triangle is computed
            max_corr, shift = compute_pair_correlation(all_series[m], all_series[n], max_shift, T_SIGMA, T_DELTA)

            # Convert the shift into years
            shift_years = (shift * T_DELTA.total_seconds()) / (365.25 * 24 * 3600)

            # Store the results in the matrices
            max_corr_matrix[m, n] = max_corr
            shift_matrix[m, n] = shift_years

    # Normalize
    max_corr_matrix /= max_corr_matrix.max()

    # Plot the resulting correlation matrices and scatter plot
    plot_correlation_matrices(max_corr_matrix, shift_matrix)

    return max_corr_matrix, shift_matrix


if __name__ == "__main__":
    max_corr_matrix, shift_matrix = main()
