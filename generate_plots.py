import numpy as np
from typing import Tuple


def generate_data(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simulated temperature readings from two sensors.

    Parameters
    ----------
    seed : int
        Seed for the random number generator. Use the last 4 digits of your Drexel ID if desired.

    Returns
    -------
    timestamps : numpy.ndarray
        1-D array of shape (200,) (float64) containing timestamps uniformly sampled from 0 to 10 seconds, sorted in ascending order.
    sensor_a : numpy.ndarray
        1-D array of shape (200,) (float64) containing Sensor A temperature readings (mean=25.0, std=3.0).
    sensor_b : numpy.ndarray
        1-D array of shape (200,) (float64) containing Sensor B temperature readings (mean=27.0, std=4.5).

    Notes
    -----
    The function uses ``np.random.default_rng(seed)`` for reproducible sampling and sorts the
    returned arrays by timestamp to make them ready for plotting versus time.
    """
    rng = np.random.default_rng(seed)
    n = 200

    # timestamps from 0 to 10 seconds
    timestamps = rng.uniform(0.0, 10.0, n)

    # Sensor readings (Celsius)
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n)

    # Sort by time for plotting convenience
    idx = np.argsort(timestamps)
    return timestamps[idx].astype(np.float64), sensor_a[idx].astype(np.float64), sensor_b[idx].astype(np.float64)

__all__ = ["generate_data"]
