from datetime import datetime, timedelta

from pandas import read_pickle

from energy_explorer.es_explorer import (find_acceleration_peaks,
                                         query_capacity_series)
from energy_explorer.objects import CapacitySeries
from energy_explorer.paths import ENERGY_STORAGE_CLEANED_PATH
from energy_explorer.plotters import TimeSeriesPlotter


if __name__ == "__main__":

    # User-defined fixed parameters
    FUEL_TYPE = "Solar", "Battery"
    GROUP_BY = "facility_zipcode"
    N_GROUPS = 5

    # Hardcoded parameters
    SECTOR = "Residential"
    EXCLUSIVE = True
    SELECT_FROM = "highest"

    # Smoothing parameters
    T_DELTA = timedelta(days=1)
    T_SIGMA = timedelta(days=60)

    # Define date range limits
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

    # Initialize plotter with two axes
    fuel_label = FUEL_TYPE if isinstance(FUEL_TYPE, str) else " \\& ".join(FUEL_TYPE)
    plotter = TimeSeriesPlotter(n_axes=2)
    plotter.set(0, ylabel="Capacity [MW]", title=f"{fuel_label} Capacity")
    plotter.set(1, ylabel="Acceleration [MW / day$^2$]", xlabel="Approval Date")

    # Find the overall min and max datetimes for smoothing range within the filtered data
    global_start = max(top_groups.approval_date.min().to_pydatetime(), START_DATE)
    global_end = min(top_groups.approval_date.max().to_pydatetime(), END_DATE)

    # Plot each group's data
    for n, group_id in enumerate(top_groups[GROUP_BY].unique()):
        group_data = top_groups[top_groups[GROUP_BY] == group_id].sort_values(by="approval_date")
        group_color = f"C{n}"

        # Calculate cumulative capacity
        date = group_data.approval_date.to_numpy()
        capacity = group_data.nameplate_capacity.cumsum().to_numpy() / 1000  # Convert to MW

        # Create the series
        capacity_series = CapacitySeries(time=date, capacity=capacity)

        # Smooth the capacity series using the specified start, end, delta, and sigma parameters
        capacity_series.smooth(start=global_start, end=global_end, delta=T_DELTA, sigma=T_SIGMA)

        # Plot data
        plotter.plot_series(0, capacity_series.time, capacity_series.capacity, color=group_color, label=f"{group_id}")
        plotter.plot_series(
            1, capacity_series.time, capacity_series.acceleration * 900 / T_DELTA.days**2, color=group_color
        )

        # Identify peaks in the acceleration data
        acceleration_peaks = find_acceleration_peaks(capacity_series, sigma=T_SIGMA)

        # Plot vertical lines for maxima and minima
        for peak_time, peak_value in [acceleration_peaks.max, acceleration_peaks.min]:
            if not isinstance(peak_time, float):  # Avoid NaN times
                plotter.axes[1].axvline(peak_time, color=group_color, linewidth=3, alpha=0.5)

    # Display
    plotter.show()
