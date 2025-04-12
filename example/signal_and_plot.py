import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from visualize.simulation_plotter import SimulationPlotter
from signal_edit.sampler import Sampler

# create input signal
input_points = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [2, 1],
    [2, 0],
    [10, 0]
], dtype=np.float64)

time, input_signal = Sampler.create_periodical(
    input_points,
    start_time=0.0,
    end_time=10.0,
    sampling_interval=0.1
)


# plot the signal
plotter = SimulationPlotter()

plotter.append_sequence(input_signal)

plotter.assign("input_signal", column=0, row=0, position=(0, 0),
               x_sequence=time, label="input_signal", line_style="--",
               marker='.')

plotter.plot("signal")

plt.show()
