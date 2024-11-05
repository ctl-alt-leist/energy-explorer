"""This module is useful for reading CSV files into various data structures."""

from dataclasses import fields, make_dataclass
from datetime import datetime
from typing import List, Tuple, Type, Union
from warnings import catch_warnings, simplefilter

from numpy import array, genfromtxt, recarray
from pandas import DataFrame, read_csv


ColumnType = Union[List[Tuple[str, Type]], Type]


def frame_to_array(df: DataFrame) -> recarray:
    """
    Convert a pandas DataFrame to a numpy recarray.
    """
    return df.to_records(index=False).view(recarray)


def frame_to_dataclasses(df: DataFrame, dataclass_type: Type) -> List:
    """
    Convert a pandas DataFrame to a list of dataclass instances.
    """
    return [dataclass_type(**row) for row in df.to_dict(orient="records")]


def load_csv_dataframe(file_path: str, columns: ColumnType = None, tzinfo=None) -> DataFrame:
    """
    Load CSV file using pandas, handling datetime columns and making them timezone-aware.
    """
    # Process column definitions
    column_formats = _process_column_formats(columns)
    column_names = [name for name, _ in column_formats]

    # Load CSV, skip the first row (header), and use custom column names
    df = read_csv(
        file_path,
        header=None,
        skiprows=1,
        names=column_names,
        parse_dates=[name for name, dtype in column_formats if dtype == datetime],
    )

    # Set timezone for datetime columns
    for name, dtype in column_formats:
        if dtype == datetime:
            df[name] = df[name].apply(lambda dt: dt.replace(tzinfo=tzinfo) if dt.tzinfo is None else dt)

    return df


def load_csv_array(file_path: str, columns: ColumnType = None, tzinfo=None) -> recarray:
    """
    Load CSV file using numpy, handling datetime columns and making them timezone-aware.
    """
    # Process column definitions
    column_formats = _process_column_formats(columns)
    column_names = [name for name, _ in column_formats]

    # Define dtype for structured array with custom column names
    dtype = [(name, "datetime64" if dtype == datetime else dtype) for name, dtype in column_formats]

    # Suppress ConversionWarning during genfromtxt call and skip the header row
    with catch_warnings():
        simplefilter("ignore", category=UserWarning)
        data = genfromtxt(
            file_path,
            delimiter=",",
            skip_header=1,
            names=column_names,
            dtype=dtype,
            encoding="utf-8",
            invalid_raise=False,
        )

    # Convert to record array and apply timezone to datetime fields
    rec_data = data.view(recarray)
    for name, dtype in column_formats:
        if dtype == datetime:
            times = array([datetime.fromisoformat(str(d)).replace(tzinfo=tzinfo) for d in rec_data[name]])
            rec_data[name] = times

    return rec_data


def load_csv_dataclasses(file_path: str, columns: ColumnType, tzinfo=None) -> List:
    """
    Load CSV file using numpy, convert each row into a dataclass instance, handling datetime and timezone awareness.
    """
    # Process column definitions
    column_formats = _process_column_formats(columns)

    # Define dtype for structured array
    dtype = [(name, "datetime64" if dtype == datetime else dtype) for name, dtype in column_formats]

    # Suppress ConversionWarning during genfromtxt call
    with catch_warnings():
        simplefilter("ignore", category=UserWarning)
        data = genfromtxt(file_path, delimiter=",", names=True, dtype=dtype, encoding="utf-8", invalid_raise=False)

    # Prepare data as list of dataclass instances
    dataclass_type = columns if isinstance(columns, type) else make_dataclass("DynamicDataClass", column_formats)
    instances = []
    for row in data:
        row_data = {}
        for (name, dtype), value in zip(column_formats, row):
            if dtype == datetime:
                dt = datetime.fromisoformat(str(value)).replace(tzinfo=tzinfo)
                row_data[name] = dt
            else:
                row_data[name] = value
        instances.append(dataclass_type(**row_data))

    return instances


def _process_column_formats(columns: ColumnType) -> List[Tuple[str, Type]]:
    """
    Process column definitions, returning a list of column names and types.
    """
    if isinstance(columns, type) and hasattr(columns, "__dataclass_fields__"):
        return [(field.name, field.type) for field in fields(columns)]
    return columns
