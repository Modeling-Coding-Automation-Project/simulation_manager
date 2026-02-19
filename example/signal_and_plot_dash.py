"""
This script demonstrates the generation and visualization of
 a pulse signal using custom signal processing and plotting utilities.
It creates a pulse input signal with specified parameters,
 then visualizes the signal using a Plotly/Dash-based simulation plotter.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from visualize.simulation_plotter_dash import SimulationPlotterDash
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
plotter = SimulationPlotterDash(activate_dump=True)

plotter.append_sequence_name(input_signal, "input_signal")

plotter.assign("input_signal", column=0, row=0, position=(0, 0),
               x_sequence=time, label="input_signal", line_style="--",
               marker='.')
plotter.assign_all("input_signal", position=(1, 0),
                   x_sequence=time, label="input_signal")

plotter.plot("signal")
