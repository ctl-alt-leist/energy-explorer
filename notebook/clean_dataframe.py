from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from pandas import DataFrame
from pgeocode import Nominatim

from energy_explorer.readers import load_csv_dataframe


@dataclass
class EnergyStorage:
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


def clean_energy_storage_dataframe(frame: DataFrame) -> DataFrame:
    """
    Cleans the Energy Storage DataFrame by processing each row, updating fuel types, adding geo_coords, and modifying
    the DataFrame in place.

    Args:
        frame (DataFrame): The initial uncleaned DataFrame.
    """
    print("Starting full cleaning process for the energy storage DataFrame...", flush=True)
    print("Tasks: Process fuel types - Add geo-coordinates", flush=True)

    # Process each row for fuel types
    n = len(frame)
    for m, row in frame.iterrows():
        progress_line("Processing rows", m, n)
        frame.at[m, "fuel_types"] = sort_fuel_types(frame.at[m, "fuel_types"])

    # Add geo coordinates for unique zipcodes
    add_geo_coords_to_frame(frame)

    print("...done.")

    return frame


def add_geo_coords_to_frame(frame: DataFrame) -> DataFrame:
    """
    Adds a geo_coords column to the DataFrame by finding coordinates for unique zip codes and mapping them.

    Args:
        frame: The DataFrame containing energy storage data.
    """
    # Initialize Nominatim for geolocation
    nomi = Nominatim("us")

    # Extract unique zipcodes and set up dictionary for mapping
    unique_zipcodes = frame.facility_zipcode.unique()
    n = len(unique_zipcodes)

    zip_to_coords = {}
    for m, zipcode in enumerate(unique_zipcodes):
        progress_line("Mapping geo-coordinates", m, n)
        location = nomi.query_postal_code(str(zipcode))
        zip_to_coords[zipcode] = (
            (location.latitude, location.longitude) if location.latitude and location.longitude else (None, None)
        )

    # Map coordinates to the DataFrame
    frame.geo_coords = frame.facility_zipcode.map(zip_to_coords)

    return frame


def sort_fuel_types(fuel_types: str) -> List[str]:
    """
    Processes and formats a fuel type entry by splitting on delimiters, capitalizing each component, and recombining
    them with a space separator.

    Args:
        fuel_types: A string which is intended to be a list of fuel types, delimited by "_ ".

    Returns:
        List[str]: A list of cleaned and formatted fuel type strings.
    """
    formatted_fuel_types = []

    # Split and process fuel types
    fuel_types_rep = fuel_types.replace("_ ", "_").replace(" & ", "_").replace("/", " / ")

    for fuel_type in fuel_types_rep.split("_"):
        fuel_type_capitalized = " ".join([word.capitalize() for word in fuel_type.split()])
        fuel_type_cleaned = fuel_type_capitalized.replace(" / ", "/").replace(" Pv", "")
        formatted_fuel_types.append(fuel_type_cleaned)

    return formatted_fuel_types


def progress_line(description, m, n) -> None:
    """Prints a clean progress line."""
    p = 100 * (m + 1) / n
    print(f"{description}: {p:.0f}% of {n} entries.", end="\r", flush=True)
    print("\033[K", end="\r", flush=True) if p == 100 else None


if __name__ == "__main__":
    from energy_explorer.paths import (ENERGY_STORAGE_CLEANED_PATH,
                                       ENERGY_STORAGE_PATH)

    # Load and clean the data
    es_frame = load_csv_dataframe(ENERGY_STORAGE_PATH, EnergyStorage)
    es_frame_cleaned = clean_energy_storage_dataframe(es_frame)

    # Write the clean data back
    es_frame_cleaned.to_pickle(ENERGY_STORAGE_CLEANED_PATH)
