from cartopy.crs import Orthographic, PlateCarree
from cartopy.feature import BORDERS, COASTLINE, STATES, NaturalEarthFeature
from matplotlib.pyplot import show, subplots
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pandas import Series


class TimeSeriesPlotter:
    def __init__(self, n_axes: int = 1) -> None:
        """Initializes a TimeSeriesPlotter instance for plotting time series data with multiple axes."""
        self.fig, self.axes = subplots(n_axes, 1, figsize=(9 + n_axes, 3 + 2 * n_axes), sharex=True)
        if n_axes == 1:
            self.axes = [self.axes]
        self.setup()

    def plot_series(self, i: int, time: Series, data: Series, **kwargs) -> None:
        """Plots a given variable against time on a specified axes."""
        self.axes[i].plot(time, data, **kwargs)
        if "label" in kwargs:
            self.axes[i].legend()

    def fill_between_series(
        self, i: int, time: Series, lower: Series, upper: Series, color: str = "lightgrey", **kwargs
    ) -> None:
        """Fills the area between two time series on a specified axes."""
        self.axes[i].fill_between(time, lower, upper, color=color, **kwargs)

    def set(self, index: int, **kwargs) -> None:
        """Passes arguments to `pyplot.ax.set()` for a specified axes."""
        self.axes[index].set(**kwargs)

    def setup(self) -> None:
        """Configure the axes."""
        for ax in self.axes:
            ax.minorticks_on()
            ax.grid(True)

    def clear(self, index: int) -> None:
        """Clears the specified axes but retains the x-axis datetime range."""
        xlim = self.axes[index].get_xlim()
        self.axes[index].clear()
        self.axes[index].set_xlim(xlim)
        self.axes[index].minorticks_on()
        self.axes[index].grid(True)

    def show(self) -> None:
        """Displays the plot."""
        show()


class MapPlotter:
    def __init__(self, central_longitude: float = -119.5, central_latitude: float = 37.5):
        """Initialize an EnergyStorageMap instance with a base map centered at specified coordinates."""
        self.fig, self.ax = subplots(
            figsize=(9, 8),
            subplot_kw={
                "projection": Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
            },
        )
        self.gl = None
        self._setup_base_map()

    def _setup_base_map(self):
        """Set up the base map with essential features and gridlines."""
        # Set title
        self.ax.set(title="California Energy Storage System")

        # Define the limits in (lat, lon)
        self.ax.set_extent([-126, -112, 30, 44], crs=PlateCarree())

        # Add map features
        self.ax.add_feature(COASTLINE)
        self.ax.add_feature(BORDERS, linestyle=":")
        self.ax.add_feature(STATES)
        self.ax.add_feature(
            NaturalEarthFeature("cultural", "urban_areas", scale="10m", edgecolor="gray", facecolor="none"),
            linestyle="--",
            linewidth=0.5,
        )

        # Initialize gridlines (default on)
        self._add_gridlines()

    def set_projection_center(self, central_longitude: float, central_latitude: float):
        """Set the center of the map projection and update the plot."""
        self.ax.projection = Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
        self.ax.set_extent([-126, -112, 30, 44], crs=PlateCarree())
        self.fig.canvas.draw_idle()

    def _add_gridlines(self, show: bool = True, lon_lines: int = 2, lat_lines: int = 2):
        """Add gridlines to the map and set their intervals.

        Args:
            show: Whether to show or hide gridlines.
            lon_lines: The interval between longitude gridlines.
            lat_lines: The interval between latitude gridlines.
        """
        if self.gl:
            self.gl.visible = show
        else:
            self.gl = self.ax.gridlines(draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--")
            self.gl.xlocator = MaxNLocator(integer=True)
            self.gl.ylocator = MaxNLocator(integer=True)
            self.gl.top_labels = False
            self.gl.right_labels = False
        self.fig.canvas.draw_idle()

    def toggle_gridlines(self, show: bool, x_interval: int = 2, y_interval: int = 2):
        """Toggle the visibility of gridlines and set their intervals.

        Args:
            show: True to show gridlines, False to hide them.
            x_interval: The interval between longitude gridlines.
            y_interval: The interval between latitude gridlines.
        """
        if self.gl:
            self.gl.visible = show
        else:
            self._add_gridlines(show=show, lon_lines=x_interval, lat_lines=y_interval)
        self.fig.canvas.draw_idle()

    def plot_coords(self, coords: NDArray[float], color: str = "tab:blue", marker: str = "o", **kwargs):
        """
        Plot (lat, lon) coordinates on the existing map.

        Args:
            coords: (N, 2) array of latitude and longitude pairs.
            color: The color of the coordinate points.
            marker: Marker style for coordinate points.
        """
        self.ax.plot(*coords.T[::-1], marker, color=color, **kwargs, transform=PlateCarree())

    def plot_cities(self, city_coords: NDArray[float], color: str = "tab:red", marker: str = "^", markersize: int = 4):
        """
        Plot city coordinates on the existing map.

        Args:
            city_coords: (N, 2) array of latitude and longitude pairs for cities.
            color: The color of the city markers.
            marker: Marker style for city markers.
            markersize: Marker size for city markers.
        """
        self.ax.plot(*city_coords.T[::-1], marker, color=color, markersize=markersize, transform=PlateCarree())

    def show_map(self):
        """Display the map."""
        show()
