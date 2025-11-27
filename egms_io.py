"""I/O helpers to build EGMS ortho deliverables.

The module focuses on reading/writing CSV+XML bundles, packaging them as
ZIP archives, and emitting simplified raster exports used by the QC
tooling. Logging hooks are provided so the caller can integrate the
functions inside larger pipelines.
"""

import hashlib
import logging
import zipfile
import zlib
from io import SEEK_SET, BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal, Optional, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine

from metadata import generate_xml_content  # type: ignore[import-not-found]


@overload
def read_csv_or_zip(
    path: Path,
    *,
    read_xml: Literal[True] = ...,
    logger: Optional[logging.Logger] = ...,
    **kwargs: Any,
) -> tuple[gpd.GeoDataFrame, str]:
    ...


@overload
def read_csv_or_zip(
    path: Path,
    *,
    read_xml: Literal[False],
    logger: Optional[logging.Logger] = ...,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    ...


def read_csv_or_zip(
    path: Path,
    *,
    read_xml: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, str]:
    """Load an EGMS CSV from disk or from a ZIP archive.

    The function inspects coordinate columns to build a GeoDataFrame and
    optionally reads the companion XML payload.

    Args:
        path: CSV file or ZIP archive containing exactly one CSV member.
        read_xml: When ``True`` (default) loads the sibling XML file and
            returns it alongside the table.
        logger: Optional logger; ``logging.getLogger(__name__)`` is used
            when absent.
        **kwargs: Additional ``pandas.read_csv`` keyword arguments. The
            ``columns`` alias is supported to request a subset.

    Returns:
        Either the GeoDataFrame alone or a ``(GeoDataFrame, xml_text)``
        tuple depending on ``read_xml``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    path = Path(path).resolve()
    logger.debug("Reading from %s", path)

    usecols = kwargs.pop("columns", None)
    nrows = kwargs.pop("nrows", None)
    logger.debug("pandas.read_csv kwargs: %s", kwargs)
    suffix = path.suffix.lower()
    csv_member: Optional[str] = None
    xml_member: Optional[str] = None
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zipf:
            csv_members = [n for n in zipf.namelist() if n.endswith(".csv")]
            if not csv_members:
                msg = f"No CSV file found inside {path}"
                raise FileNotFoundError(msg)
            csv_member = csv_members[0]
            xml_candidates = [n for n in zipf.namelist() if n.endswith(".xml")]
            xml_member = xml_candidates[0] if xml_candidates else None
            with zipf.open(csv_member) as csvfile:
                header = pd.read_csv(csvfile, nrows=0, **kwargs)
    elif suffix == ".csv":
        header = pd.read_csv(path, nrows=0, **kwargs)
    else:
        msg = f"Path extension {path.suffix} not recognised"
        raise NotImplementedError(msg)

    kwargs["nrows"] = nrows
    if {"easting", "northing"}.issubset(header.columns):
        geomcols = ["easting", "northing"]
        epsg = 3035
    elif {"longitude", "latitude"}.issubset(header.columns):
        geomcols = ["longitude", "latitude"]
        epsg = 4326
    else:
        msg = "Input path has neither lat/lon nor east/north columns"
        raise ValueError(msg)
    logger.debug("Detected EPSG:%d via columns %s", epsg, geomcols)

    if usecols is not None:
        kwargs["usecols"] = list(set(geomcols + usecols) - {"geometry"})
    logger.debug("Using columns: %s", kwargs.get("usecols", "all columns"))

    xml_content: Optional[str] = None
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zipf:
            assert csv_member is not None
            with zipf.open(csv_member) as csvfile:
                df = pd.read_csv(csvfile, **kwargs)
            if read_xml:
                if xml_member is None:
                    msg = f"No XML file found inside {path}"
                    raise FileNotFoundError(msg)
                with zipf.open(xml_member) as xml_file:
                    xml_content = xml_file.read().decode("utf-8")
    else:
        df = pd.read_csv(path, **kwargs)
        if read_xml:
            with path.with_suffix(".xml").open("r", encoding="utf-8") as file:
                xml_content = file.read()

    selected_cols = list(df.columns if usecols is None else usecols)
    selected_cols = [col for col in selected_cols if col != "geometry"]
    geodf = gpd.GeoDataFrame(
        data=df[selected_cols],
        geometry=gpd.points_from_xy(df[geomcols[0]], df[geomcols[1]], crs=epsg),
        crs=epsg,
    )

    if read_xml:
        assert xml_content is not None  # for typing clarity
        return geodf, xml_content
    return geodf


XmlContent = str | bytes | bytearray | memoryview


def write_csv_or_zip(
    path: Path,
    csv_content: pd.DataFrame,
    xml_content: Optional[XmlContent] = None,
    *,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> None:
    """Persist the provided table as CSV or as a ZIP bundle.

    Args:
        path: Destination path (``.csv`` or ``.zip``).
        csv_content: DataFrame to serialise via ``to_csv``.
        xml_content: Optional XML payload to embed in the ZIP file or to
            write next to the CSV path.
        logger: Optional logger; ``logging.getLogger(__name__)`` is used
            when absent.
        **kwargs: Extra arguments forwarded to ``DataFrame.to_csv``. The
            special keys ``csv_name`` and ``xml_name`` define the member
            names when creating a ZIP archive.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    path = Path(path)
    logger.debug("Writing to %s", path)
    csv_name = kwargs.pop("csv_name", path.with_suffix(".csv").name)
    xml_name = kwargs.pop("xml_name", path.with_suffix(".xml").name)
    logger.debug("Using internal CSV name: %s", csv_name)
    logger.debug("Using internal XML name: %s", xml_name)
    kwargs.update({"index": False, "chunksize": 100_000})
    logger.debug("DataFrame.to_csv kwargs: %s", kwargs)

    # If given and is bytes, convert XML content
    if xml_content is None:
        xml_payload: Optional[str] = None
    elif isinstance(xml_content, (bytes, bytearray, memoryview)):
        xml_payload = bytes(xml_content).decode()
    else:
        xml_payload = xml_content

    match path.suffix:
        case ".zip":
            with zipfile.ZipFile(
                path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=zlib.Z_BEST_COMPRESSION,
            ) as ziph:
                if xml_payload is not None:
                    ziph.writestr(xml_name, xml_payload)
                csv_stringio = StringIO()
                csv_content.to_csv(csv_stringio, **kwargs)
                csv_stringio.seek(0, SEEK_SET)
                csv_payload = csv_stringio.read()
                ziph.writestr(csv_name, csv_payload)
        case ".csv":
            csv_content.to_csv(path, **kwargs)
            if xml_payload is not None:
                xml_path = path.with_suffix(".xml")
                with xml_path.open("w", encoding="utf-8") as handle:
                    handle.write(xml_payload)
        case _:
            msg = f"Path extension {path.suffix} not recognised"
            raise NotImplementedError(msg)


def ortho_writer(
    df: Optional[gpd.GeoDataFrame],
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create the EGMS ZIP+GeoTIFF deliverables for the provided data.

    The CSV data (minus geometry) is stored inside a ZIP archive together
    with a freshly generated metadata XML. Additionally, a GeoTIFF and a
    MD5 checksum file are produced next to the ZIP.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if df is None or df.empty:
        logger.warning("No data to write for product %s", output_path.stem)
        return

    xml_content = generate_xml_content()

    # Ensure output is a ZIP file
    output_path = output_path.with_suffix(".zip")
    if output_path.exists():
        logger.warning(
            "Output file %s already exists and will be overwritten", output_path
        )
    write_csv_or_zip(
        output_path,
        df.drop(columns="geometry"),
        xml_content=xml_content,
        logger=logger,
    )
    logger.debug("Wrote ZIP to %s", output_path)

    tiff_data = create_ortho_geotiff(
        df, "mean_velocity", logger=logger
    )
    if tiff_data is not None:
        output_path.with_suffix(".tiff").write_bytes(tiff_data)
        logger.debug("Wrote GeoTIFF to %s", output_path.with_suffix(".tiff"))
    else:
        logger.warning("GeoTIFF creation skipped for product %s", output_path.stem)

    checksum = hashlib.md5(output_path.read_bytes())
    output_path.with_suffix(".md5").write_text(
        f"{checksum.hexdigest()}  {output_path.name}\n"
    )
    logger.debug("Wrote MD5 checksum to %s", output_path.with_suffix(".md5"))


def create_ortho_geotiff(
    df: gpd.GeoDataFrame,
    mean_velocity_field: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[bytes]:
    """Rasterize the points-based mean velocity field into a GeoTIFF.

    Args:
        df: Input GeoDataFrame.
        mean_velocity_field: Column to burn into the raster.
        logger: Optional logger; ``logging.getLogger(__name__)`` is used
            when absent.

    Returns:
        The GeoTIFF payload as bytes, or ``None`` when the raster cannot
        be created.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if df.empty:
        logger.warning("Input GeoDataFrame is empty")
        return None

    if mean_velocity_field not in df:
        logger.warning("No mean velocity field found in CSV")
        return None

    df = df.to_crs(3035)
    min_e = np.floor(df.geometry.x.min() / 100_000) * 100_000
    max_n = np.ceil(df.geometry.y.max() / 100_000) * 100_000
    logger.debug("Raster bounds: min_e=%f, max_n=%f", min_e, max_n)

    meta: dict[str, Any] = dict(
        driver="GTiff",
        height=1_000,
        width=1_000,
        count=1,
        dtype=np.float32,
        crs=3035,
        nodata=-9999,
        transform=Affine.translation(min_e, max_n)
        * Affine.scale(100.0, -100.0),
    )
    shapes = (
        (geom, value)
        for geom, value in zip(df.geometry, df[mean_velocity_field])
    )
    burned = features.rasterize(
        shapes=shapes,
        fill=meta["nodata"],
        out_shape=(meta["width"], meta["height"]),
        transform=meta["transform"],
        all_touched=True,
    )
    logger.debug("Rasterized data with shape %s", burned.shape)
    stream = BytesIO()
    with rasterio.open(stream, "w+", **meta) as out:
        out.write(burned, indexes=1)
    logger.debug("Wrote GeoTIFF data to in-memory stream")

    return stream.getvalue()