import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from visualize.simulation_plotter import SimulationPlotter
from signal_edit.sampler import Sampler, PulseGenerator

# create input signal
input_points = PulseGenerator.generate_pulse_points(
    start_time=1.0,
    period=2.0,
    pulse_width=50.0,
    pulse_amplitude=1.0,
    duration=10.0,
    number_of_pulse=1
)

time, input_signal = Sampler.create_periodical(
    input_points,
    start_time=0.0,
    end_time=10.0,
    sampling_interval=0.1
)


# plot the signal
plotter = SimulationPlotter()

plotter.append_sequence_name(input_signal, "input_signal")

plotter.assign("input_signal", column=0, row=0, position=(0, 0),
               x_sequence=time, label="input_signal", line_style="--",
               marker='.')

plotter.plot("signal")

plt.show()
