"""
IDE-only exports. Runtime imports still come from the installed jax_fdm package.

Plotter is a thin *args/**kwargs wrapper, so the stub points at the COMPAS
class that actually owns figsize and the rest of the constructor.
"""

from compas_plotter.plotter import Plotter as Plotter
from jax_fdm.visualization.plotters.loss_plotter import LossPlotter as LossPlotter
from jax_fdm.visualization.viewers.viewer import Viewer as Viewer
