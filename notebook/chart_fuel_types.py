from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.pyplot import Figure, ion, show, subplots
from numpy import nan, triu_indices_from, unique, zeros_like
from pandas import DataFrame, read_pickle
from seaborn import heatmap

from energy_explorer.es_explorer import get_capacity_distribution


def fuel_capacity_chart(frame: DataFrame, exclusive: bool = True, shared: bool = False, bin_size: int = None) -> Figure:
    """
    Creates a heatmap of mean capacity values for exclusive pairs of fuel types in entries with 1 or 2 types.

    Args:
        frame: The cleaned DataFrame containing cumulative capacities and fuel types.
        exclusive: If True, selects only entries containing exactly the fuel types in fuel_type.
        shared: If True, divides the entry capacity by the number of fuel types in each entry.
        bin_size: Optional; if provided, chunks the color mapping into bins of the specified size.
    """
    # Filter for entries with 1 or 2 fuel types
    subset_frame = frame[frame.fuel_types.apply(len) <= 2]

    # Get the unique fuel types
    fuel_types = unique(subset_frame.explode("fuel_types").fuel_types)

    # Initialize a matrix to store the mean capacities for each pair of fuel types
    capacity_matrix = DataFrame(nan, index=fuel_types, columns=fuel_types)

    # Calculate the capacity mean for each pair of fuel types
    for i, fuel_type_i in enumerate(fuel_types):
        for j, fuel_type_j in enumerate(fuel_types[i:], start=i):
            # Set up the filter condition
            fuel_list = [fuel_type_i, fuel_type_j] if fuel_type_i != fuel_type_j else [fuel_type_i]
            dist = get_capacity_distribution(subset_frame, fuel_type=fuel_list, exclusive=exclusive, shared=shared)

            # Assign the mean capacity value to the matrix
            mean_capacity = dist.capacity.mean() if len(dist.capacity) > 0 else nan
            capacity_matrix.at[fuel_type_i, fuel_type_j] = mean_capacity
            capacity_matrix.at[fuel_type_j, fuel_type_i] = mean_capacity

    # Mask the upper triangle to only show the lower triangle
    mask = zeros_like(capacity_matrix, dtype=bool)
    mask[triu_indices_from(mask, k=1)] = True

    # Define the color mapping based on bin_size
    if bin_size:
        min_capacity, max_capacity = 0.0, capacity_matrix.max().max()
        bins = list(range(int(min_capacity), int(max_capacity) + bin_size, bin_size))
        norm = BoundaryNorm(bins, ncolors=256)
    else:
        norm = Normalize(vmin=capacity_matrix.min().min(), vmax=capacity_matrix.max().max())

    # Label
    label = "Mean Shared Capacity [kW]" if shared else "Mean Capacity [kW]"

    # Plot the heatmap with the color legend as specified
    fig, ax = subplots(1, 1, figsize=(10, 8))
    heatmap(
        capacity_matrix,
        mask=mask,
        annot=False,
        cmap="GnBu",
        cbar_kws={"label": label},
        ax=ax,
        norm=norm,
        xticklabels=True,
        yticklabels=True,
    )

    return fig


def fuel_approval_chart(frame: DataFrame, exclusive: bool = False, bin_size: int = 1) -> Figure:
    """
    Creates a heatmap showing the number of approvals by year for each fuel type.

    Args:
        frame: The cleaned DataFrame containing approval dates and fuel types.
        exclusive: If True, selects only entries containing exactly the specified fuel types.
        bin_size: The bin size for color scaling in the heatmap legend.

    Returns:
        Figure: The heatmap figure.
    """
    # Convert approval_date to period for yearly grouping
    approval_year = frame.approval_date.dt.to_period("Y")
    fuel_types = frame.fuel_types.explode()

    # Count approvals by year and fuel type
    approval_counts = (
        DataFrame({"approval_year": approval_year, "fuel_type": fuel_types})
        .groupby(["fuel_type", "approval_year"])
        .size()
        .unstack(fill_value=0)
    )

    # Define the color mapping based on bin_size
    min_count, max_count = 0, approval_counts.max().max()

    # Adjust bin_size to fit within colormap limits
    max_bins = 256
    adjusted_bin_size = max(bin_size, (max_count - min_count) // max_bins + 1)
    bins = list(range(min_count, max_count + adjusted_bin_size, adjusted_bin_size))
    norm = BoundaryNorm(bins, ncolors=max_bins)

    # Mask zero values to show them as white
    mask = approval_counts == 0

    # Create the heatmap
    fig, ax = subplots(1, 1, figsize=(12, 10))
    heatmap(
        approval_counts,
        mask=mask,
        cmap="YlGnBu",
        cbar_kws={"label": "Number of Approvals"},
        ax=ax,
        annot=False,
        norm=norm,
    )

    # Configure axes
    ax.set(xlabel=None, ylabel=None, title="Number of Approvals by Fuel Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return fig


if __name__ == "__main__":
    from energy_explorer.paths import ENERGY_STORAGE_CLEANED_PATH

    # Load the cleaned data
    energy_storage = read_pickle(ENERGY_STORAGE_CLEANED_PATH)

    # Turn ion on
    ion()

    if True:
        # Get the shared capacity chart
        fig_a = fuel_capacity_chart(energy_storage, exclusive=True, shared=False, bin_size=50)

    if False:
        # Generate the heatmap
        fig_b = fuel_approval_chart(energy_storage, exclusive=True)

    # Show
    show()
