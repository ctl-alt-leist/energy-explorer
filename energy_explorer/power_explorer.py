from dataclasses import dataclass
from datetime import datetime
from typing import Optional


if __name__ == "__main__":
    from energy_explorer.readers import load_csv_dataframe

    frame = load_csv_dataframe("./data/steel_industry_data.csv", EnergyLoad)
