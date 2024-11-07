from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from matplotlib.pyplot import Figure, setp, show, subplots
from numpy import histogram2d, where
from pandas import DataFrame

from energy_explorer.readers import load_csv_dataframe


@dataclass
class EnergyLoad:
    timestamp: datetime  # Continuous-time data taken on the first of the month [date]
    real_energy: float  # Industry energy consumption [kWh]
    reactive_energy_lagging: float  # Lagging current reactive power [kVarh]
    reactive_energy_leading: float  # Leading current reactive power [kVarh]
    co2_emission: float  # CO2 emissions [ppm]
    power_factor_lagging: float  # Lagging power factor [/1.0]
    power_factor_leading: float  # Leading power factor [/1.0]
    nsm: int  # Number of seconds from midnight [s]
    weekday_status: int  # Week status (Weekend (0) or a Weekday (1))
    day_of_week: str  # Day of the week (Sunday, Monday, ... , Saturday)
    load_type: str  # Load type (Light Load, Medium Load, Maximum Load)
    month: Optional[int]  # Month from timestamp [1-12]


def clean_power_data(frame: DataFrame) -> DataFrame:
    """Clean and prepare the power data by sorting by timestamp and normalizing power factor."""
    # Sort by timestamp
    frame = frame.sort_values(by="timestamp")

    # Normalize power factor to 1.0 scale
    frame.power_factor_lagging /= 100.0
    frame.power_factor_leading /= 100.0

    # Compute month
    frame["month"] = frame.timestamp.dt.month

    return frame


def plot_power_time_series(frame: DataFrame) -> Figure:
    """Plot the time series data of real energy, reactive energy, emissions, and power factor in a shared figure."""
    fig, axes = subplots(3, 1, figsize=(11, 8), sharex=True)

    # Plot Real Energy
    axes[0].plot(frame.timestamp, frame.real_energy, color="tab:blue")
    axes[0].set(ylabel="Real Energy [kWh]")

    # Plot Reactive Energy (Lagging and Leading)
    axes[1].plot(frame.timestamp, frame.reactive_energy_lagging, color="tab:green", label="Lagging")
    axes[1].plot(frame.timestamp, frame.reactive_energy_leading, color="tab:red", label="Leading")
    axes[1].set(ylabel="Reactive Energy [kVarh]")
    axes[1].legend()

    # Plot CO2 Emissions
    axes[2].plot(frame.timestamp, frame.co2_emission, color="tab:purple")
    axes[2].set(ylabel="CO$_2$ Emissions [tons]")

    # Style and show
    for ax in axes:
        setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.minorticks_on()
        ax.grid()

    return fig


def plot_co2_heatmap(frame: DataFrame, x_column: Tuple[str, str], y_column: Tuple[str, str]) -> Figure:
    """Plot a heatmap of CO2 emissions for selected energy comparisons."""
    # Select the column and units
    x_name, x_units = x_column
    y_name, y_units = y_column

    # Numpize
    x_ = frame[x_name].to_numpy()
    y_ = frame[y_name].to_numpy()
    co2_ = frame.co2_emission.to_numpy()

    # Select only the data with non-zero values
    w = where((x_ / x_.max() > 0.02) & (y_ / y_.max() > 0.02) & (co2_ / co2_.max() > 0.02))[0]
    x = x_[w]
    y = y_[w]
    co2 = co2_[w]

    # Initialize the figure and axes
    fig, ax = subplots(figsize=(8, 6))

    # Generate heatmap
    h, xedges, yedges = histogram2d(x, y, bins=50, weights=co2)
    im = ax.imshow(
        h.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    # Style the axes
    x_label = x_name.replace("_", " ").title() + f" [{x_units}]"
    y_label = y_name.replace("_", " ").title() + f" [{y_units}]"
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.minorticks_on()

    # Add minorticks and a label to the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.minorticks_on()
    cbar.set_label("CO$_2$ Emissions [tons]")

    # Figure styling
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Load the data
    filepath = "./data/steel_industry_data.csv"
    frame = load_csv_dataframe(filepath, EnergyLoad)

    # Clean and prepare the data
    frame = clean_power_data(frame)

    # Plot time series data
    if True:
        fig_a = plot_power_time_series(frame)
        fig_a.savefig("figures/steel_data/time_series_full.png")

    # Plot CO2 heatmap (select columns)
    if False:
        fig_b = plot_co2_heatmap(frame, x_column=("real_energy", "kWh"), y_column=("reactive_energy_lagging", "kVarh"))
        fig_b.savefig("figures/steel_data/co2_heat_map.png")

    show()
