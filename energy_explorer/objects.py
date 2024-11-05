from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

from numpy import (arange, argmax, argmin, array, convolve, datetime64, diff,
                   exp, interp, mean, nan, pad, timedelta64)
from numpy.typing import NDArray


@dataclass
class EnergyStorage:
    """Dataclass representing the column structure of the (cleaned) energy storage dataframe."""

    utility: str
    nameplate_capacity: float
    fuel_types: str | List[str]
    facility_city: str
    facility_county: str
    caiso_flag: str
    facility_zipcode: int
    customer_sector: str
    approval_date: datetime
    geo_coords: Optional[Tuple[float, float]] = None


@dataclass
class CapacityDistribution:
    """Dataclass representing a fuel type with associated capacities and counts for a given distribution."""

    fuel_type: Union[str, List[str]]
    capacity: NDArray[float]
    count: NDArray[int]


@dataclass
class CapacitySeries:
    """Dataclass representing a time-series of cumulative capacity."""

    time: NDArray[Union[datetime, datetime64]]
    capacity: NDArray[float]

    @property
    def acceleration(self) -> NDArray[float]:
        """
        Computes the second derivative of the capacity. The resulting array will have the same length as `time` and
        `capacity`, with the edge values padded accordingly.

        Returns:
            NDArray of the second derivative (acceleration) of the capacity with the same length as the time array.
        """
        # Calculate the second derivative using `diff`
        acceleration_ = diff(self.capacity, n=2)

        # Pad the resulting array
        acceleration = pad(acceleration_, (1, 1), mode="edge")
        if False:
            from numpy import concatenate

            acceleration = concatenate([[0], acceleration_])

        return acceleration

    def smooth(
        self, start: Union[datetime, datetime64], end: Union[datetime, datetime64], delta: timedelta, sigma: timedelta
    ) -> None:
        """
        Smooths the capacity series by interpolating data onto a new time grid and applying a Gaussian kernel.
        This method uses numpy's interp for linear interpolation (without extrapolation) and trims edge data
        based on kernel coverage, ensuring that time and capacity remain aligned.

        Args:
            start: Start datetime of the new smoothing time range.
            end: End datetime of the new smoothing time range.
            delta: Time step size for the new grid.
            sigma: Standard deviation of the Gaussian kernel as timedelta.
        """
        # Convert start and end to numpy datetime64 for uniformity
        start_ = datetime64(start, "s")
        end_ = datetime64(end, "s")

        # Generate new time grid within the specified range
        size_ = int((end_ - start_) / timedelta64(int(delta.total_seconds()), "s"))
        new_times = array(
            [start_ + i * timedelta64(int(delta.total_seconds()), "s") for i in range(size_)],
            dtype="datetime64[s]",
        )

        # Linear interpolation using numpy's interp (without extrapolation)
        new_times_float = new_times.astype("datetime64[s]").astype(float)
        original_times_float = self.time.astype("datetime64[s]").astype(float)
        interpolated_capacity = interp(new_times_float, original_times_float, self.capacity)

        # Define Gaussian kernel based on sigma
        kernel_radius = int(3 * sigma.total_seconds() / delta.total_seconds())
        kernel_range = arange(
            -kernel_radius * delta.total_seconds(), (kernel_radius + 1) * delta.total_seconds(), delta.total_seconds()
        )
        kernel = exp(-0.5 * (kernel_range / sigma.total_seconds()) ** 2)
        kernel /= kernel.sum()  # Normalize the kernel

        # Convolve with kernel and trim edge data based on kernel radius
        smoothed_capacity = convolve(interpolated_capacity, kernel, mode="same")
        trimmed_times = new_times[kernel_radius:-kernel_radius]
        trimmed_capacity = smoothed_capacity[kernel_radius:-kernel_radius]

        # Ensure all capacity values are non-negative
        trimmed_capacity[trimmed_capacity < 0] = 0

        # Update instance attributes with aligned time and capacity arrays
        self.time = trimmed_times
        self.capacity = trimmed_capacity


@dataclass
class AccelerationPeaks:
    """
    Dataclass for storing acceleration peak information.

    Attributes:
        minima: NDArray of shape (N, 2) containing (time, peak_value) for each detected minimum.
        maxima: NDArray of shape (N, 2) containing (time, peak_value) for each detected maximum.
    """

    minima: NDArray[float]
    maxima: NDArray[float]

    @property
    def max(self) -> tuple[Union[datetime, float], float]:
        """Returns the (time, value) of the maximum value in the maxima peaks, or (nan, nan) if no valid maxima."""
        if self.maxima.ndim == 2 and self.maxima.shape[1] == 2 and len(self.maxima) > 0:
            m = argmax(self.maxima[:, 1])
            return self.maxima[m, 0], self.maxima[m, 1]
        return nan, nan

    @property
    def min(self) -> tuple[Union[datetime, float], float]:
        """Returns the (time, value) of the minimum value in the minima peaks, or (nan, nan) if no valid minima."""
        if self.minima.ndim == 2 and self.minima.shape[1] == 2 and len(self.minima) > 0:
            m = argmin(self.minima[:, 1])
            return self.minima[m, 0], self.minima[m, 1]
        return nan, nan

    @property
    def frequency(self) -> float:
        """
        Computes the temporal frequency of peaks in [1/year], using the mean difference
        in time between consecutive peaks. Handles missing peaks by averaging over available data.
        """
        if len(self.maxima) < 2:
            return 0.0

        time_diffs = diff(self.maxima[:, 0]).astype("timedelta64[D]").astype(int)
        avg_time_diff = mean(time_diffs)

        return 365.25 / avg_time_diff if avg_time_diff > 0 else 0.0
