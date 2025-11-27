"""Estimate displacement statistics and coherence metrics."""

import logging
import re
from datetime import date, datetime
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd

from model import Model


FrameLike = pd.DataFrame | gpd.GeoDataFrame
ArrayFloat = npt.NDArray[np.floating[Any]]


def wrap_values_to_interval(
    values: ArrayFloat,
    period: float = 2 * np.pi,
) -> ArrayFloat:
    """Wrap values into the symmetric interval ``[-period/2, period/2)``.

    Args:
        values: Input numeric array to wrap.
        period: Length of the wrapping period (defaults to ``2*pi``).

    Returns:
        Array with the same shape as ``values`` containing wrapped data.
    """

    half_period = period / 2.0
    return ((values + half_period) % period) - half_period


def model_estimation(
    df: Optional[FrameLike],
    wave_length_mm: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[FrameLike]:
    """Estimate model-based stats for each point in a (Geo)DataFrame.

    The function expects one column per acquisition whose name matches
    ``YYYYMMDD``. The first chronological column becomes the temporal origin
    (t = 0). For each row the function fits, in sequence, cubic, quadratic, and
    linear models with sinusoidal terms leveraging ``Model``. It then returns a
    new (Geo)DataFrame appending the following statistics to the input:

    - ``rmse``: Root mean square error of the cubic fit.
    - ``seasonality`` / ``seasonality_std``: amplitude and uncertainty of the
      sinusoidal component.
    - ``acceleration`` / ``acceleration_std``: double the quadratic term and
      its uncertainty.
    - ``mean_velocity`` / ``mean_velocity_std``: linear term and uncertainty.
    - ``temporal_coherence``: coherence derived from the linear fit residuals
      after converting phase to millimetres using ``wave_length_mm``.

    Args:
        df: DataFrame or GeoDataFrame where each row is a point and date-like
            columns store displacement values.
        wave_length_mm: Sensor wavelength expressed in millimetres, used to
            convert phase residuals when computing coherence.
        logger: Optional logger for progress and debug messages.

    Returns:
        The input DataFrame (or GeoDataFrame) copy augmented with the fields
        listed above, or ``None`` when ``df`` is ``None``/empty.

    Notes:
        - Date columns must be numeric strings in the YYYYMMDD format.
        - Existing statistic columns will be overwritten.
        - Logging remains silent unless a logger propagates INFO/DEBUG.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if df is None or df.empty:
        logger.warning("Input dataframe is None or empty, returning None.")
        return None

    logger.info("Starting model_estimation.")
    dates_str: list[str] = [
        col for col in df.columns if re.fullmatch(r"\d+", col)
    ]
    logger.debug("Identified %d date columns.", len(dates_str))
    if len(dates_str) == 0:
        logger.warning(
            "No date columns found in the dataframe, returning input as is."
        )
        return df
    dates: list[date] = [
        datetime.strptime(col, "%Y%m%d").date() for col in dates_str
    ]
    anchor_date = dates[0]
    logger.debug("Using anchor_date: %s.", anchor_date)

    timeline: ArrayFloat = (
        np.array([(x - anchor_date).days for x in dates]).astype(float) / 365.0
    )
    logger.debug("Timeline created with %d points.", timeline.size)

    # Use a contiguous array for performance.
    # The shape is (n_dates, n_points)
    data: ArrayFloat = np.ascontiguousarray(df.loc[:, dates_str].values.T)
    # Cubic model fit: c3, c2, c1, cos, sin, const
    model = Model(degree=3, sinusoid=True, bias=True)
    model.fit(timeline, data)
    logger.debug("Cubic model fitted.")
    # Make sure that the model fitted intersects the origin
    df.loc[:, dates_str] -= model.constant[:, None]
    # Initialize new fields once to avoid repeated dataframe assignments.
    new_fields: dict[str, ArrayFloat] = {}
    # Evaluate the model on the timeline (shape is (n_dates, n_points))
    predicted: ArrayFloat = model.eval(timeline)
    new_fields["rmse"] = np.sqrt(np.nanmean((data - predicted) ** 2, axis=0))
    new_fields["seasonality"] = np.sqrt(model.cosine ** 2 + model.sine ** 2)
    t_mod: ArrayFloat = np.vstack(
        [
            timeline,
            timeline**2,
            timeline**3,
            np.cos(2.0 * np.pi * timeline) - 1,
            np.sin(2.0 * np.pi * timeline),
            np.ones_like(timeline),
        ]
    ).astype(float)
    cov_mtx: ArrayFloat = np.linalg.pinv(t_mod @ t_mod.T)
    sinusoidal_var: float = float(cov_mtx[3, 3] + cov_mtx[4, 4])
    if np.allclose(sinusoidal_var, 0.0):
        logger.warning(
            "Sinusoidal variance is zero, setting seasonality_std to NaN."
        )
        new_fields["seasonality_std"] = 0.0
    else:
        new_fields["seasonality_std"] = (
            np.sqrt(((4.0 - np.pi) / 2.0) * (sinusoidal_var / 2.0))
            * new_fields["rmse"]
        )
    logger.debug("Cubic model statistics computed.")
    # Quadratic model fit: c2, c1, cos, sin, const
    model = Model(degree=2, sinusoid=True, bias=True)
    model.fit(timeline, data)
    new_fields["acceleration"] = 2 * model.quadratic
    predicted = model.eval(timeline)
    rms: ArrayFloat = np.sqrt(np.nanmean((data - predicted) ** 2, axis=0))
    mean: ArrayFloat = np.nanmean((data - predicted), axis=0)
    std_dev: ArrayFloat = np.sqrt(rms**2 + mean**2)
    t_mod[1, :] *= 0.5  # Adjust for acceleration term
    t_mod = t_mod[(0, 1, 3, 4, 5), :]
    cov_mtx = np.linalg.pinv(t_mod @ t_mod.T)
    new_fields["acceleration_std"] = np.sqrt(cov_mtx[1, 1]) * std_dev
    logger.debug("Parabolic model statistics computed.")
    # fit del modello lineare: c, cos, sin, const
    model = Model(degree=1, sinusoid=True, bias=True)
    model.fit(timeline, data)
    new_fields["mean_velocity"] = model.linear
    predicted = model.eval(timeline)
    rms = np.sqrt(np.nanmean((data - predicted) ** 2, axis=0))
    mean = np.nanmean((data - predicted), axis=0)
    std_dev = np.sqrt(rms**2 + mean**2)
    # Recompute covariance matrix for the linear model design (t, cos, sin, const)
    t_mod = t_mod[(0, 2, 3, 4), :]
    cov_mtx = np.linalg.pinv(t_mod @ t_mod.T)
    new_fields["mean_velocity_std"] = np.sqrt(cov_mtx[0, 0]) * std_dev
    logger.debug("Linear model statistics computed.")
    rad_to_mm = wave_length_mm / (4 * np.pi)
    # Coerenza temporale da modello lineare
    c: ArrayFloat = model.linear[None, :]
    dt: ArrayFloat = timeline[:, None]
    const: ArrayFloat = model.constant[None, :]
    phase: ArrayFloat = wrap_values_to_interval(
        (data - (c * dt + const)) / rad_to_mm
    )
    cohe: ArrayFloat = np.abs(np.exp(phase * 1j).mean(axis=0))
    new_fields["temporal_coherence"] = cohe
    logger.debug("Temporal coherence computed.")

    # Assign all new fields at once
    df = df.copy().assign(**new_fields)
    logger.info("Model estimation completed.")

    return df
