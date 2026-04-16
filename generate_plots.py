import numpy as np
import matplotlib.pyplot as plt
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


def plot_scatter(ax: plt.Axes, timestamps: np.ndarray, sensor_a: np.ndarray, sensor_b: np.ndarray) -> None:
    """Plot sensor temperature readings as a scatter plot on the provided Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Existing matplotlib Axes object to draw the scatter plot on. The function modifies this
        Axes in place and returns None.
    timestamps : numpy.ndarray
        1-D array of shape (200,) with timestamps in seconds. Typically sorted ascending.
    sensor_a : numpy.ndarray
        1-D array of shape (200,) of Sensor A temperature readings (Celsius).
    sensor_b : numpy.ndarray
        1-D array of shape (200,) of Sensor B temperature readings (Celsius).

    Returns
    -------
    None
        The function updates the provided Axes object in place and does not return a value.

    Notes
    -----
    This function does not call ``plt.show()`` so callers can compose multiple plots or adjust
    figure-level settings before displaying or saving.
    """
    # Basic validation (fail early with informative messages)
    if timestamps.ndim != 1 or sensor_a.ndim != 1 or sensor_b.ndim != 1:
        raise ValueError("timestamps, sensor_a, and sensor_b must be 1-D arrays")
    if not (timestamps.shape[0] == sensor_a.shape[0] == sensor_b.shape[0]):
        raise ValueError("timestamps, sensor_a, and sensor_b must have the same length")

    ax.scatter(timestamps, sensor_a, s=25, alpha=0.7, label='Sensor A', color='C0', marker='o')
    ax.scatter(timestamps, sensor_b, s=25, alpha=0.7, label='Sensor B', color='C1', marker='x')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Sensor Temperature Readings')
    ax.legend()
    ax.grid(True)


__all__ = ["generate_data", "plot_scatter"]
