"""Utilities for renaming and formatting EGMS GeoDataFrames."""

from dataclasses import dataclass
from datetime import datetime
import itertools
import logging
import re
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from base_encoding import encode


@dataclass
class FieldSpec:
    """Specification for a single output field."""

    name: str
    previous_name: Optional[str] = None
    precision: Optional[int] = None


@dataclass
class OutputNamingConvention:
    """Definition of the naming and precision constraints."""

    field_list: list[FieldSpec]
    date_format: str
    date_precision: int


def format_dataframe(
    df: gpd.GeoDataFrame | None,
    naming_convention: OutputNamingConvention,
    logger: Optional[logging.Logger] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Return a GeoDataFrame shaped as required by ``naming_convention``.

    Parameters
    ----------
    df : geopandas.GeoDataFrame | None
        Source GeoDataFrame that needs to be reformatted.
    naming_convention : OutputNamingConvention
        Naming and precision requirements for both scalar and date columns.
    logger : logging.Logger | None
        Optional logger to be used for diagnostic messages.

    Returns
    -------
    geopandas.GeoDataFrame | None
        The reformatted GeoDataFrame or ``None`` when the input is ``None`` or
        empty.
    """
    if logger is None:
        logger = logging.getLogger("format_dataframe")

    if df is None:
        logger.info("The input dataframe is None. Returning None.")
        return None

    # df is a GeoDataFrame at this point
    if df.empty:
        logger.info("The input dataframe is empty. Returning None.")
        return None

    # Put aside the geometry column to preserve it
    geom_orig = df.geometry
    # Change the column names by means of the provided naming convention
    dates_cols_str: list[str] = [
        col for col in df.columns if re.fullmatch(r"\d+", col)
    ]
    logger.debug("Identified %d date columns.", len(dates_cols_str))
    if len(dates_cols_str) == 0:
        logger.warning(
            "No date columns found in the dataframe, returning input as is."
        )
        return df
    dates_cols = [
        datetime.strptime(col, "%Y%m%d").date()
        for col in dates_cols_str
    ]
    dates_convention = {
        old_col_name: date_col.strftime(naming_convention.date_format)
        for date_col, old_col_name in zip(dates_cols, dates_cols_str)
    }
    logger.debug("Date columns renaming: %s", dates_convention)
    # Create geometric columns if they are required by the naming convention
    names_set = {spec.name for spec in naming_convention.field_list}
    geom_cols = {"latitude", "longitude"}
    if names_set.intersection(geom_cols):
        geom = df.to_crs("EPSG:4326").geometry
        df["longitude"] = geom.x
        df["latitude"] = geom.y
        logger.debug("Created latitude and longitude columns from geometry.")
    geom_cols = {"easting", "northing"}
    if names_set.intersection(geom_cols):
        geom = df.to_crs("EPSG:3035").geometry
        df["easting"] = geom.x
        df["northing"] = geom.y
        logger.debug("Created easting and northing columns from geometry.")
    renames = {
        spec.previous_name: spec.name
        for spec in naming_convention.field_list
        if spec.previous_name is not None
    }
    logger.debug("Renaming columns: %s", renames)
    df = df.rename(
        columns={
            **renames,
            **dates_convention,
        }
    )
    # If some columns are missing, an error will be raised
    all_necessary_cols = list(
        itertools.chain(
            [spec.name for spec in naming_convention.field_list],
            sorted(dates_convention.values()),
        )
    )
    logger.debug("All necessary columns: %s", all_necessary_cols)
    for col in all_necessary_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing after renaming")
    columns_to_drop = df.columns.difference(all_necessary_cols)
    if not columns_to_drop.empty:
        logger.info("Dropping columns: %s", columns_to_drop.tolist())
        df = df.drop(columns=columns_to_drop)
    # Sort columns according to the naming convention
    df = df[all_necessary_cols]
    logger.debug(
        "DataFrame after renaming and reordering:\n%s",
        df.head(),
    )
    # Fix each field precision according to the data precision model
    int_field_conversion: dict[str, str] = {}
    float_field_precision: dict[str, int] = {}
    for field in naming_convention.field_list:
        match field.precision:
            case None | "":
                pass
            case 0:
                # integer
                int_field_conversion[field.name] = "int32"
            case _:
                # float
                float_field_precision[field.name] = field.precision
    if naming_convention.date_precision == 0:
        int_field_conversion = {
            **int_field_conversion,
            **{col: "int32" for col in dates_cols_str},
        }
    else:
        float_field_precision = {
            **float_field_precision,
            **{col: naming_convention.date_precision for col in dates_cols_str},
        }
    logger.debug("Integer field conversions: %s", int_field_conversion)
    logger.debug("Float field precisions: %s", float_field_precision)
    df = df.astype(int_field_conversion)
    df = df.round(float_field_precision)

    formatted_geodf = gpd.GeoDataFrame(
        df,
        geometry=geom_orig,
        crs=geom_orig.crs,
    )
    logger.info(
        "Formatted GeoDataFrame:\n%s",
        formatted_geodf.head(),
    )
    return formatted_geodf


def compute_pid(df: gpd.GeoDataFrame) -> pd.Series:
    """Compute the PID string from the ``northing``/``easting`` columns.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        GeoDataFrame containing ``northing`` and ``easting`` columns.

    Returns
    -------
    pandas.Series
        Series with the PID strings encoded in base-62.
    """
    num = (df.northing / 100).astype(np.int64) * 2**32 + (
        df.easting / 100
    ).astype(np.int64)
    pid = "1" + num.apply(encode, p=62, pad=9)
    return pid