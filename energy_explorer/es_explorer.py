from datetime import timedelta
from typing import List, Union

from numpy import array, greater, less, percentile, zeros
from pandas import DataFrame
from scipy.signal import argrelextrema

from energy_explorer.objects import (AccelerationPeaks, CapacityDistribution,
                                     CapacitySeries)


# GEOGRAPHY


def get_unique_zip_geo(frame: DataFrame):
    """
    Extracts unique (lat, lon) coordinates from the DataFrame mapped to their zipcodes.

    Args:
        frame: The DataFrame containing facility zipcodes and geo-coordinates.

    Returns:
        Tuple of two numpy arrays: zipcodes and geo_coords.
    """
    unique_zipcodes = frame.facility_zipcode.drop_duplicates().index
    unique_geo_df = frame.loc[unique_zipcodes, frame.geo_coords]
    unique_geo = array([(geo[0], geo[1]) for geo in unique_geo_df])

    return frame.loc[unique_zipcodes, frame.facility_zipcode].values, unique_geo


# FUEL TYPES & CAPACITY


def get_capacity_distribution(
    frame: DataFrame, fuel_type: Union[str, List[str]], exclusive: bool = False, shared: bool = True
) -> CapacityDistribution:
    """
    Creates a capacity distribution for specified fuel types with optional exclusive filtering and shared capacity.

    Args:
        frame: The cleaned DataFrame containing cumulative capacities and fuel types.
        fuel_type: A single fuel type as a string or a list of fuel types.
        exclusive: If True, selects only entries containing exactly the fuel types in fuel_type.
        shared: If True, divides the entry capacity by the number of fuel types in each entry.

    Returns:
        CapacityDistribution: Distribution containing filtered capacities and counts for the specified fuel type(s).
    """
    selected_entries = query_fuel_types(frame, fuel_type, exclusive)

    # Calculate shared or net capacity based on the shared argument
    if shared:
        capacity_values = selected_entries.nameplate_capacity / selected_entries.fuel_types.apply(len)
    else:
        capacity_values = selected_entries.nameplate_capacity

    # Determine counts for each selected entry
    fuel_counts = selected_entries.fuel_types.apply(len) if shared else zeros(len(selected_entries), dtype=int) + 1

    return CapacityDistribution(fuel_type=fuel_type, capacity=capacity_values, count=fuel_counts)


def query_fuel_types(frame: DataFrame, fuel_type: Union[str, List[str]], exclusive: bool = False) -> DataFrame:
    """
    Filters the DataFrame based on fuel type combinations.

    Args:
        frame: The DataFrame containing cumulative capacities and fuel types.
        fuel_type: A single fuel type or list of fuel types for filtering.
        exclusive: If True, selects only entries containing exactly the fuel types.

    Returns:
        A filtered DataFrame based on the specified fuel type combinations.
    """
    if isinstance(fuel_type, str):
        fuel_filter = [fuel_type]
    else:
        fuel_filter = fuel_type

    if exclusive:
        required_types = set(fuel_filter)
        condition = frame.fuel_types.apply(lambda types: set(types) == required_types)
    else:
        condition = frame.fuel_types.apply(lambda types: any(ft in types for ft in fuel_filter))

    return frame[condition]


def query_capacity_series(
    es_frame: DataFrame,
    fuel_type: str | List[str],
    exclusive: bool = True,
    sector: str = "Residential",
    group_by: str = "facility_zipcode",
    n_groups: int = None,
    select_from: str = "highest",
) -> DataFrame:
    """
    Selects and returns a filtered DataFrame based on fuel type, sector, and group-based cumulative capacity.

    Args:
        es_frame: The DataFrame containing energy storage data.
        fuel_types: A single fuel type or list of fuel types for filtering.
        exclusive: If True, selects only entries containing exactly the fuel types.
        sector: The customer sector to filter by (e.g., 'Residential').
        group_by: The column used to group data (e.g., 'facility_zipcode').
        n_groups: Number of groups to return based on cumulative capacity.
        select_from: Determines if the function returns the "highest", "middle", or "lowest" n_groups.

    Returns:
        DataFrame: A filtered DataFrame including entries for selected groups based on rated capacity.
    """
    # Filter data for the specified sector
    es_frame_sel = es_frame[es_frame.customer_sector == sector]

    # Filter data based on fuel types
    es_frame_sel = query_fuel_types(es_frame_sel, fuel_type, exclusive=exclusive)

    # Identify cumulative capacity by group
    capacity_sum_by_group = es_frame_sel.groupby(group_by).nameplate_capacity.sum().sort_values()

    # Select groups based on n_groups if specified
    if n_groups is not None:
        if select_from.lower() == "highest":
            selected_groups = capacity_sum_by_group.nlargest(n_groups).index
        elif select_from.lower() == "lowest":
            selected_groups = capacity_sum_by_group.nsmallest(n_groups).index
        elif select_from.lower() == "middle":
            mid_start = (len(capacity_sum_by_group) - n_groups) // 2
            selected_groups = capacity_sum_by_group.iloc[mid_start : mid_start + n_groups].index
        else:
            raise ValueError("select_from must be one of 'highest', 'middle', or 'lowest'")

        # Make the selection, and sort them by the datetime
        selected_frame = es_frame_sel[es_frame_sel[group_by].isin(selected_groups)]
    else:
        # Return all groups if n_groups is None
        selected_frame = es_frame_sel

    # Sort the selected frame by approval date
    selected_frame = selected_frame.sort_values(by="approval_date", ascending=True)

    # Remove duplicate approval dates by keeping the first occurrence
    selected_frame = selected_frame[~selected_frame.duplicated(subset="approval_date", keep="first")]

    return selected_frame


if __name__ == "__main__":
    from pandas import read_pickle

    from energy_explorer.paths import ENERGY_STORAGE_CLEANED_PATH

    es_frame = read_pickle(ENERGY_STORAGE_CLEANED_PATH)

    es_frame_sel = query_capacity_series(
        es_frame=es_frame,
        fuel_type=["Solar"],
        exclusive=True,
        sector="Residential",
        group_by="facility_zipcode",
        n_groups=5,
        select_from="lowest",
    )


def find_acceleration_peaks(
    capacity_series: CapacitySeries,
    sigma: timedelta,
) -> AccelerationPeaks:
    """
    Identify isolated positive and negative peaks in the acceleration of a smoothed CapacitySeries object.
    Peaks are defined as local maxima/minima that meet a threshold based on the 95th percentile of nearby values.

    Args:
        capacity_series: The CapacitySeries object containing time and capacity (already smoothed).
        sigma: The sigma used in the Gaussian smoothing kernel, used to determine the minimum distance between peaks.

    Returns:
        AccelerationPeaks: A dataclass containing arrays of minima and maxima peaks.
    """
    # Convert sigma to an integer number of points based on time spacing
    t_interval = (capacity_series.time[1] - capacity_series.time[0]).astype("timedelta64[s]").item().total_seconds()
    t_width = 3 * int(sigma.total_seconds() / t_interval)

    # Get acceleration data
    acceleration = capacity_series.acceleration

    # Find indices of relative maxima and minima in acceleration
    max_indices = argrelextrema(acceleration, comparator=greater, order=t_width)[0]
    min_indices = argrelextrema(acceleration, comparator=less, order=t_width)[0]

    # Helper function to extract 95th percentile peak values in a window around each peak
    def get_peaks(indices, percentile_target):
        peak_times = []
        peak_values = []

        for n in indices:
            start = max(0, n - t_width)
            end = min(len(acceleration), n + t_width)
            peak_val = percentile(acceleration[start:end], percentile_target)

            peak_times.append(capacity_series.time[n])
            peak_values.append(peak_val)

        return array(list(zip(peak_times, peak_values)))

    # Calculate maxima and minima using the 95th percentile for maxima and 5th percentile for minima
    maxima = get_peaks(max_indices, 95)
    minima = get_peaks(min_indices, 5)

    return AccelerationPeaks(minima=minima, maxima=maxima)
