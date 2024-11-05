from copy import deepcopy
from datetime import datetime, timedelta

from matplotlib.pyplot import show, subplots, tight_layout
from numpy import clip, dot, isinf, isnan, nanmax, where, zeros
from numpy.linalg import norm
from numpy.typing import NDArray
from pandas import read_pickle

from energy_explorer.es_explorer import query_capacity_series
from energy_explorer.objects import CapacitySeries
from energy_explorer.paths import ENERGY_STORAGE_CLEANED_PATH
from energy_explorer.plotters import TimeSeriesPlotter


def compute_cosine_similarity(series_m: CapacitySeries, series_n: CapacitySeries) -> float:
    """
    Compute the cosine similarity of the acceleration vectors of two CapacitySeries objects.

    Args:
        series_m: The first CapacitySeries object.
        series_n: The second CapacitySeries object.

    Returns:
        float: Cosine similarity value between the acceleration vectors of series_m and series_n.
    """
    accel_m = series_m.acceleration
    accel_n = series_n.acceleration

    # Ensure both series are of the same length
    min_len = min(len(accel_m), len(accel_n))
    accel_m = accel_m[:min_len]
    accel_n = accel_n[:min_len]

    # Compute the dot product and norms
    dot_product = dot(accel_m, accel_n)
    norm_m = norm(accel_m)
    norm_n = norm(accel_n)

    return dot_product / (norm_m * norm_n + 1e-9)


def predict_capacity_series(
    series_m: CapacitySeries, series_n: CapacitySeries, similarity_coeff: float
) -> CapacitySeries:
    """
    Predict the capacity of the n series using the acceleration of the m series, scaled by the given similarity
    coefficient.

    Args:
        series_m: The reference CapacitySeries object.
        series_n: The CapacitySeries object to be modified for prediction.
        similarity_coeff: The cosine similarity coefficient between the two series.

    Returns:
        CapacitySeries: The predicted CapacitySeries object.
    """
    # Deepcopy the n series for modification
    series_p = deepcopy(series_n)

    # Set the second half of the n_series_modified capacity to zero
    midpoint = len(series_p.capacity) // 2
    series_p.capacity[midpoint:] = 0

    # Cap the similarity coefficient within a reasonable range to avoid overflow
    similarity_coeff = clip(similarity_coeff, -1.0, 1.0)

    # Use m's capacity to predict n's future capacity
    for t in range(midpoint - 1, len(series_p.capacity) - 1):
        # Compute the fractional change in capacity for series m
        dc_m = series_m.capacity[t + 1] - series_m.capacity[t]

        # Compute the change in capacity for the predicted series based on similarity coefficient
        dc_p = 2.0 * dc_m * similarity_coeff**2

        # Update the capacity for the predicted series
        predicted_capacity = series_p.capacity[t] + dc_p

        # Handle potential overflow, invalid values, or NaNs
        if isnan(predicted_capacity) or isinf(predicted_capacity):
            predicted_capacity = 0

        series_p.capacity[t + 1] = clip(predicted_capacity, -1e9, 1e9)

    return series_p


def plot_similarity_matrix(similarity_matrix: NDArray[float], title: str = "Cosine Similarity Matrix") -> None:
    """Plot the similarity matrix."""
    fig, ax = subplots(figsize=(7, 6))
    ax.set(title=title)
    cax = ax.imshow(similarity_matrix, cmap="viridis", aspect="auto")
    fig.colorbar(cax)
    tight_layout()
    show()


def main() -> NDArray[float]:
    """Compute cosine similarity of capacity series accelerations."""

    # User-defined fixed parameters
    FUEL_TYPE = "Solar"
    EXCLUSIVE = False
    SECTOR = "Residential"
    SELECT_FROM = "highest"
    GROUP_BY = "facility_zipcode"
    N_GROUPS = 50  # `None` selects all groups

    # Smoothing parameters
    T_DELTA = timedelta(days=1)
    T_SIGMA = timedelta(days=30)
    START_DATE = datetime(2001, 1, 1)
    END_DATE = datetime(2025, 1, 1)

    # Load the data
    es_frame = read_pickle(ENERGY_STORAGE_CLEANED_PATH)

    # Filter data to only include entries within the specified date range
    es_frame = es_frame[(es_frame.approval_date >= START_DATE) & (es_frame.approval_date < END_DATE)]

    # Query data for the top groups based on cumulative capacity
    top_groups = query_capacity_series(
        es_frame=es_frame,
        fuel_type=FUEL_TYPE,
        exclusive=EXCLUSIVE,
        sector=SECTOR,
        group_by=GROUP_BY,
        n_groups=N_GROUPS,
        select_from=SELECT_FROM,
    )

    # Prepare and normalize each capacity series
    all_series = []
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

    # Create the similarity matrix
    n_series = len(all_series)
    similarity_matrix = zeros((n_series, n_series))

    # Calculate the cosine similarity for each pair of series
    for m in range(n_series):
        for n in range(m + 1):  # Only compute for lower triangle (n <= m)
            similarity_score = compute_cosine_similarity(all_series[m], all_series[n])
            similarity_matrix[m, n] = similarity_score
            similarity_matrix[n, m] = similarity_score  # Symmetric matrix
    similarity_matrix = similarity_matrix / (similarity_matrix.diagonal()[:, None] + 1e-9)

    # Find the highest similarity coefficient below 0.99
    highest_similarity = similarity_matrix[similarity_matrix < 0.99].max()

    # Get the indices (m, n) for the highest similarity value
    indices = where(similarity_matrix == highest_similarity)
    m, n = int(indices[0][0]), int(indices[1][0])

    series_m = all_series[m]
    series_n = all_series[n]
    series_p = predict_capacity_series(series_m, series_n, highest_similarity)

    # Bounds calculation
    alpha = 1.0 - 0.5 * similarity_score**2

    # Compute upper and lower bounds that grow with time
    capacity_p_upper = series_p.capacity + alpha * series_p.capacity / 2.0
    capacity_p_lower = series_p.capacity - alpha * series_p.capacity / 2.0

    # Plot the resulting similarity matrix
    f_label = FUEL_TYPE if isinstance(FUEL_TYPE, str) else " & ".join(FUEL_TYPE)
    plot_similarity_matrix(similarity_matrix, title=f"{f_label} Similarity")

    # Plot the predicted series
    plotter = TimeSeriesPlotter(1)
    plotter.set(0, ylabel="Capacity [MW]")
    plotter.plot_series(0, series_m.time, series_m.capacity, color="black", label="Reference")
    plotter.plot_series(0, series_n.time, series_n.capacity, color="blue", label="True")
    plotter.plot_series(0, series_p.time, series_p.capacity, color="blue", linestyle="--", label="Predicted")
    plotter.fill_between_series(
        0, series_p.time, capacity_p_lower, capacity_p_upper, color="blue", alpha=0.25, label="Predictive Bounds"
    )
    plotter.set(0, title=f"Similarity Score: {similarity_score:.3f}")
    plotter.show()

    return similarity_matrix


if __name__ == "__main__":
    similarity_matrix = main()
