"""
This script demonstrates the generation and visualization of
 a pulse signal using custom signal processing and plotting utilities.
It creates a pulse input signal with specified parameters,
 then visualizes the signal using a simulation plotter.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from visualize.simulation_plotter import SimulationPlotter
from signal_edit.sampler import Sampler, PulseGenerator

# create input signal
time, input_signal = PulseGenerator.sample_pulse(
    sampling_interval=0.1,
    start_time=1.0,
    period=2.0,
    pulse_width=50.0,
    pulse_amplitude=1.0,
    duration=10.0,
    number_of_pulse=1
)

# plot the signal
plotter = SimulationPlotter(activate_dump=True)

plotter.append_sequence_name(input_signal, "input_signal")

plotter.assign("input_signal", column=0, row=0, position=(0, 0),
               x_sequence=time, label="input_signal", line_style="--",
               marker='.')

plotter.plot("signal")

plt.show()
