"""Polynomial-plus-sinusoid model fitting helpers."""

import numpy as np


class Model:
    """
    Model class for fitting and evaluating polynomial models with optional sinusoidal and bias (constant) terms.

    This class allows fitting a polynomial model (up to degree 3) to multiple time series, with optional sinusoidal (cosine and sine) and constant (bias) terms. The model can then be evaluated on arbitrary time grids.

    Attributes:
        degree (int): Degree of the polynomial (1 to 3).
        sinusoid (bool): Whether to include sinusoidal (cosine and sine) terms.
        bias (bool): Whether to include a constant (bias) term.
    """

    def __init__(self, degree: int, sinusoid: bool, bias: bool):
        """
        Initialize the Model.

        Args:
            degree (int): Degree of the polynomial (must be 0, 1, 2, or 3).
            sinusoid (bool): If True, include sinusoidal (cosine and sine) terms.
            bias (bool): If True, include a constant (bias) term.

        Raises:
            ValueError: If degree is not in [0, 1, 2, 3].
        """
        if not (0 <= degree <= 3):
            raise ValueError("Degree must be between 0 and 3.")
        self.degree = degree
        self.sinusoid = sinusoid
        self.bias = bias

    def fit(self, timesteps: np.ndarray, timeseries: np.ndarray):
        """
        Fit the model to the provided time series data.

        Args:
            timesteps (np.ndarray): 1D array of time points (length T).
            timeseries (np.ndarray): 2D array of shape (T, N), where each column is a time series to fit.

        Returns:
            Model: The fitted model instance (self).

        Raises:
            TypeError: If inputs are not numpy arrays.
            ValueError: If input shapes are invalid.

        The coefficients are stored in self._coeffs, with columns ordered as:
            [linear, quadratic (if degree>=2), cubic (if degree>=3), cosine (if sinusoid), sine (if sinusoid), constant (if bias)]
        """
        if not isinstance(timesteps, np.ndarray) or not isinstance(timeseries, np.ndarray):
            raise TypeError("Both timesteps and timeseries must be numpy arrays.")
        if timesteps.ndim != 1 or timeseries.ndim != 2:
            raise ValueError("Timesteps must be a 1D array and timeseries must be a 2D array.")
        if timesteps.size != timeseries.shape[0]:
            raise ValueError("Timesteps and timeseries must have the same length in the first dimension.")

        # Ensure numeric, contiguous arrays (avoids hidden copies in BLAS)
        t = np.asarray(timesteps, dtype=float)
        Y = np.ascontiguousarray(timeseries, dtype=float)

        # Guard against models with zero terms
        num_coeffs = self.degree + 2 * int(self.sinusoid) + int(self.bias)
        poly_base = np.full((t.size, num_coeffs), np.nan)
        if self.degree >= 1:
            poly_base[:, 0] = t
        if self.degree >= 2:
            poly_base[:, 1] = t ** 2
        if self.degree >= 3:
            poly_base[:, 2] = t ** 3
        if self.sinusoid:
            poly_base[:, self.degree] = np.cos(2 * np.pi * t) - 1.0
            poly_base[:, self.degree + 1] = np.sin(2 * np.pi * t)
        if self.bias:
            poly_base[:, -1] = 1.0
        matrix = np.linalg.pinv(poly_base)
        self._coeffs = matrix @ np.array(Y)
        return self
    
    @property
    def linear(self):
        """
        Get the linear coefficient(s) of the fitted model.

        Returns:
            np.ndarray: Array of linear coefficients, shape (N,).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        elif self.degree < 1:
            raise ValueError("Model degree is less than 1, no linear term available.")
        return self._coeffs[0, :]
    
    @property
    def quadratic(self):
        """
        Get the quadratic coefficient(s) of the fitted model.

        Returns:
            np.ndarray: Array of quadratic coefficients, shape (N,).

        Raises:
            ValueError: If the model has not been fitted yet or degree < 2.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        elif self.degree < 2:
            raise ValueError("Model degree is less than 2, no quadratic term available.")
        return self._coeffs[1, :]
    
    @property
    def cubic(self):
        """
        Get the cubic coefficient(s) of the fitted model.

        Returns:
            np.ndarray: Array of cubic coefficients, shape (N,).

        Raises:
            ValueError: If the model has not been fitted yet or degree < 3.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        elif self.degree < 3:
            raise ValueError("Model degree is less than 3, no cubic term available.")
        return self._coeffs[2, :]
    
    @property
    def cosine(self):
        """
        Get the cosine coefficient(s) of the fitted model (if sinusoid is True).

        Returns:
            np.ndarray: Array of cosine coefficients, shape (N,).

        Raises:
            ValueError: If the model has not been fitted yet or sinusoid is False.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        elif not self.sinusoid:
            raise ValueError("Model does not include a sinusoidal term.")
        return self._coeffs[self.degree, :]

    @property
    def sine(self):
        """
        Get the sine coefficient(s) of the fitted model (if sinusoid is True).

        Returns:
            np.ndarray: Array of sine coefficients, shape (N,).

        Raises:
            ValueError: If the model has not been fitted yet or sinusoid is False.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        elif not self.sinusoid:
            raise ValueError("Model does not include a sinusoidal term.")
        return self._coeffs[self.degree + 1, :]

    @property
    def constant(self):
        """
        Get the constant (bias) coefficient(s) of the fitted model (if bias is True).

        Returns:
            np.ndarray: Array of constant coefficients, shape (N,).

        Raises:
            ValueError: If the model has not been fitted yet or bias is False.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        elif not self.bias:
            raise ValueError("Model does not include a constant term.")
        return self._coeffs[-1, :]
    
    @property
    def num_models(self):
        """
        Get the number of models fitted (number of columns in the input time series).

        Returns:
            int: Number of models fitted.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        return self._coeffs.shape[1]

    def eval(self, timesteps) -> np.ndarray:
        """
        Evaluate the fitted model(s) on a given timeline.

        Args:
            timesteps (np.ndarray): 1D array of time points to evaluate the model(s) on.

        Returns:
            np.ndarray: Array of shape (T, N) with the evaluated model(s), where T is the number of time steps and N is the number of models.

        Raises:
            TypeError: If timesteps is not a numpy array.
            ValueError: If timesteps is not 1D or model is not fitted.

        Note:
            Coefficients must be expressed in the same time unit as the timeline; no conversion is performed.
        """
        if not isinstance(timesteps, np.ndarray):
            raise TypeError("Timesteps must be a numpy array.")
        if timesteps.ndim != 1:
            raise ValueError("Timesteps must be a 1D array.")
        if not hasattr(self, '_coeffs'):
            raise ValueError("Model has not been fitted yet.")
        out = np.zeros((timesteps.size, self.num_models), dtype=float)
        if self.bias:
            out += self.constant[None, :]
        if self.degree >= 1:
            out += self.linear[None, :] * timesteps[:, None]
        if self.degree >= 2:
            out += self.quadratic[None, :] * (timesteps[:, None] ** 2)
        if self.degree >= 3:
            out += self.cubic[None, :] * (timesteps[:, None] ** 3)
        if self.sinusoid:
            out += self.cosine[None, :] * (np.cos(2 * np.pi * timesteps[:, None]) - 1)
            out += self.sine[None, :] * np.sin(2 * np.pi * timesteps[:, None])
        return out
