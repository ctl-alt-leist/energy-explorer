from numpy import array
from numpy.typing import NDArray
from pandas import read_pickle
from pgeocode import Nominatim

from energy_explorer.paths import ENERGY_STORAGE_CLEANED_PATH
from energy_explorer.plotters import MapPlotter


def zipcode_to_geo(zipcodes: NDArray[int]) -> NDArray[float]:
    """
    Convert an array of zipcodes to latitude and longitude coordinates using pgeocode.

    Args:
        zipcodes: Array of zipcodes.

    Returns:
        NDArray[float]: An array of (lat, lon) pairs.
    """
    nomi = Nominatim("us")
    coordinates = []

    for zipcode in zipcodes:
        location = nomi.query_postal_code(str(zipcode))
        if location.latitude and location.longitude:
            coordinates.append((location.latitude, location.longitude))
        else:
            coordinates.append((None, None))

    return array(coordinates)


if __name__ == "__main__":
    # Load the cleaned data
    es_frame = read_pickle(ENERGY_STORAGE_CLEANED_PATH)

    # Filter data by sector "Residential" before all other operations
    es_frame_res = es_frame[es_frame.customer_sector == "Residential"]

    # Extract and plot all unique zip codes
    unique_zipcodes = es_frame_res.facility_zipcode.drop_duplicates().values
    unique_geo = zipcode_to_geo(unique_zipcodes)

    # Initialize the map plotter, and plot the full set of unique zipcodes
    energy_map = MapPlotter()
    energy_map.plot_coords(unique_geo, color="blue", marker="o", markersize=1, alpha=0.25)

    # Calculate total capacity by city within and extract the top 5 cities
    top_cities = es_frame_res.groupby("facility_city").nameplate_capacity.sum().sort_values(ascending=False).head(5)

    # Plot each of the top 5 cities
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for i, (city, _) in enumerate(top_cities.items()):
        city_data = es_frame_res[es_frame_res.facility_city == city]
        city_zipcodes = city_data.facility_zipcode.unique()
        city_geo = zipcode_to_geo(city_zipcodes)

        energy_map.plot_coords(city_geo, color=colors[i], marker="o", markersize=2, label=city)

    # Display the map with legend for top cities
    energy_map.ax.legend()
    energy_map.show_map()
