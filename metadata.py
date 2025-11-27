"""Metadata helpers shared across EGMS delivery utilities."""

from datetime import date, datetime
from typing import Optional
import xml.etree.ElementTree as ET

import geopandas as gpd
from advanced_utils.dataframes.dates import filter_date_columns


def generate_output_name(
    df: gpd.GeoDataFrame,
    direction: str,
    aoi_name: str,
    release: int,
) -> Optional[str]:
    """Create a standardized EGMS L3 product name from burst metadata.

    Args:
        df: GeoDataFrame holding the burst records that define the product.
        direction: Satellite look direction (typically ``A``/``D``; only
            the first character is used).
        aoi_name: Human-readable Area Of Interest identifier.
        release: Release counter appended at the end of the product name.

    Returns:
        The formatted product name, or ``None`` when ``df`` is empty.
    """
    if df.empty:
        return None

    dates, _ = filter_date_columns(df)
    start_date: date = min(dates)
    end_date: date = max(dates)

    return (
        f"EGMS_L3_{aoi_name}_100km_{direction:1}_"
        f"{start_date:%Y}_{end_date:%Y}_{release:1d}"
    )


def generate_xml_content() -> bytes:
    """Build a minimal L3 metadata XML snippet with static DEM/GNSS info."""
    root = ET.Element("TILE")
    ET.SubElement(root, "product_level").text = "L3"
    ET.SubElement(root, "production_facility").text = "1"
    ET.SubElement(root, "production_date").text = f"{datetime.today():%d/%m/%Y}"
    ET.SubElement(ET.SubElement(root, "dem"), "version").text = "COP-DEM_GLO-30/2020_1"
    ET.SubElement(ET.SubElement(root, "gnss"), "version").text = "2.0"

    ET.indent(root)
    return ET.tostring(
        root, method="xml", encoding="UTF-8", xml_declaration=True
    )