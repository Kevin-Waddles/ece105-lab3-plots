 import argparse import numpy as np import matplotlib.pyplot as
  plt from typing import Tuple

  def generate_data(seed: int) -> Tuple[np.ndarray, np.ndarray,
  np.ndarray]: """Generate simulated temperature readings from
  two sensors.

   Parameters
   ----------
   seed : int
       Seed for the random number generator. Use the last 4
  digits of your Drexel ID if desired.

   Returns
   -------
   timestamps : numpy.ndarray
       1-D array of shape (200,) (float64) containing timestamps
  uniformly sampled from 0 to 10 seconds, sorted in ascending
  order.
   sensor_a : numpy.ndarray
       1-D array of shape (200,) (float64) containing Sensor A 
  temperature readings (mean=25.0, std=3.0).
   sensor_b : numpy.ndarray
       1-D array of shape (200,) (float64) containing Sensor B
  temperature readings (mean=27.0, std=4.5).

   Notes
   -----
   The function uses ``np.random.default_rng(seed)`` for
  reproducible sampling and sorts the
   returned arrays by timestamp to make them ready for plotting
  versus time.
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
   return timestamps[idx].astype(np.float64),
  sensor_a[idx].astype(np.float64),
  sensor_b[idx].astype(np.float64)

  def plot_scatter(ax: plt.Axes, timestamps: np.ndarray,
  sensor_a: np.ndarray, sensor_b: np.ndarray) -> None: """Plot
  sensor temperature readings as a scatter plot on the provided
  Axes.

   Parameters
   ----------
   ax : matplotlib.axes.Axes
       Existing matplotlib Axes object to draw the scatter plot
  on. The function modifies this
       Axes in place and returns None.
   timestamps : numpy.ndarray
       1-D array of shape (200,) with timestamps in seconds.
  Typically sorted ascending.
   sensor_a : numpy.ndarray
       1-D array of shape (200,) of Sensor A temperature readings
   (Celsius).
   sensor_b : numpy.ndarray
       1-D array of shape (200,) of Sensor B temperature readings
   (Celsius).

   Returns
   -------
   None
       The function updates the provided Axes object in place and
   does not return a value.

   Notes
   -----
   This function does not call ``plt.show()`` so callers can
  compose multiple plots or adjust
   figure-level settings before displaying or saving.
   """
   # Basic validation (fail early with informative messages)
   if timestamps.ndim != 1 or sensor_a.ndim != 1 or sensor_b.ndim
   != 1:
       raise ValueError("timestamps, sensor_a, and sensor_b must
  be 1-D arrays")
   if not (timestamps.shape[0] == sensor_a.shape[0] == 
  sensor_b.shape[0]):
       raise ValueError("timestamps, sensor_a, and sensor_b must
  have the same length")

   ax.scatter(timestamps, sensor_a, s=25, alpha=0.7, 
  label='Sensor A', color='C0', marker='o')
   ax.scatter(timestamps, sensor_b, s=25, alpha=0.7, 
  label='Sensor B', color='C1', marker='x')

   ax.set_xlabel('Time (s)')
   ax.set_ylabel('Temperature (°C)')
   ax.set_title('Sensor Temperature Readings')
   ax.legend()
   ax.grid(True)

  def plot_histogram(ax: plt.Axes, sensor_a: np.ndarray,
  sensor_b: np.ndarray, bins: int = 20) -> None: """Create an
  overlaid histogram of two sensor datasets on the provided Axes.

   Parameters
   ----------
   ax : matplotlib.axes.Axes
       Matplotlib Axes to draw the histogram on. Modified in
  place.
   sensor_a : numpy.ndarray
       1-D array of sensor A temperature readings (shape (n,)).
   sensor_b : numpy.ndarray
       1-D array of sensor B temperature readings (shape (n,)).
   bins : int, optional
       Number of histogram bins to use (default: 20).

   Returns
   -------
   None
       The function updates the provided Axes and does not return
   a value.

   Notes
   -----
   The function overlays histograms for the two sensors with
  semi-transparent fills and
   draws dashed vertical lines indicating the mean of each
  distribution. It does not call
   ``plt.show()`` so callers can combine this with other
  figure-level adjustments.
   """
   # Basic validation
   if sensor_a.ndim != 1 or sensor_b.ndim != 1:
       raise ValueError("sensor_a and sensor_b must be 1-D 
  arrays")
   if sensor_a.shape[0] != sensor_b.shape[0]:
       # lengths don't strictly need to be equal for histograms, 
  but warn if different
       raise ValueError("sensor_a and sensor_b should have the
  same length")

   ax.hist(sensor_a, bins=bins, alpha=0.6, label='Sensor A', 
  color='C0')
   ax.hist(sensor_b, bins=bins, alpha=0.6, label='Sensor B',
  color='C1')

   # Mean lines
   mean_a = float(np.mean(sensor_a))
   mean_b = float(np.mean(sensor_b))
   ax.axvline(mean_a, color='C0', linestyle='--', linewidth=1, 
  label=f"A mean {mean_a:.2f}°C")
   ax.axvline(mean_b, color='C1', linestyle='--', linewidth=1, 
  label=f"B mean {mean_b:.2f}°C")

   ax.set_xlabel('Temperature (°C)')
   ax.set_ylabel('Count')
   ax.set_title('Histogram of Sensor Temperatures')
   ax.legend()
   ax.grid(True)

  def main(argv=None) -> None: """Command-line entry point that
  generates data and produces summary plots.

   Parameters
   ----------
   argv : sequence of str or None
       Optional list of command-line arguments (e.g., from
  sys.argv[1:]). If None, the
       arguments are read from the command line. Recognized
  options:

       --seed SEED    : integer RNG seed (default: 1234)
       --out PATH     : output image file path (default:
  'sensor_plots.png')
       --show         : if provided, display the figure after
  saving

   Returns
   -------
   None
       The function saves a PNG summarizing the scatter, 
  histogram, and boxplot to the
       specified output path. It may also display the figure
  interactively when ``--show``
       is supplied.
   """
   parser = argparse.ArgumentParser(description="Generate and 
  save sensor plots for simulated data")
   parser.add_argument("--seed", type=int, default=1234,
  help="RNG seed (last 4 digits of Drexel ID)")
   parser.add_argument("--out", type=str,
  default="sensor_plots.png", help="Output image file path")
   parser.add_argument("--show", action="store_true", help="Show 
  the figure interactively after saving")
   args = parser.parse_args(argv)

   timestamps, sensor_a, sensor_b = generate_data(args.seed)

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))
   plot_scatter(axes[0], timestamps, sensor_a, sensor_b)
   plot_histogram(axes[1], sensor_a, sensor_b)

   # Boxplot on third axis
   bp = axes[2].boxplot([sensor_a, sensor_b], labels=["Sensor A",
   "Sensor B"], showmeans=True, patch_artist=True)
   colors = ["C0", "C1"]
   for patch, color in zip(bp.get('boxes', []), colors):
       patch.set_facecolor(color)
   axes[2].set_ylabel('Temperature (°C)')
   axes[2].set_title('Boxplot of Sensor Temperatures')
   axes[2].grid(True, axis='y')

   fig.tight_layout()
   fig.savefig(args.out, dpi=150)
   if args.show:
       plt.show()
   else:
       plt.close(fig)

  all = ["generate_data", "plot_scatter", "plot_histogram",
  "main"]

  if name == "main": main()