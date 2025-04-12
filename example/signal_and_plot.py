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

input_signal = Sampler.create_periodical(
    input_points,
    start_time=0,
    end_time=10,
    sampling_interval=0.1
)

time = input_signal[:, 0]
value = input_signal[:, 1]


# plot the signal
plotter = SimulationPlotter()

plotter.append_sequence(value)

plotter.assign("value", column=0, row=0, position=(0, 0),
               x_sequence=time, label="value", line_style="--",
               marker='.')

plotter.plot("signal")

plt.show()
