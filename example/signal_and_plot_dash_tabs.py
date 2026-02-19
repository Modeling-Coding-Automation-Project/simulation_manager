"""
This script demonstrates the use of pre_plot() to create
multiple tabbed pages in a single Dash browser window.

Page 1 ("Pulse Signal") shows the pulse waveform.
Page 2 ("Scaled Signal") shows the same pulse scaled by 2x.
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

# create a scaled version
scaled_signal = input_signal * 2.0

# ---- set up plotter ----
plotter = SimulationPlotterDash(activate_dump=False)

# -- Page 1 --
plotter.append_sequence_name(input_signal, "input_signal")

plotter.assign("input_signal", column=0, row=0, position=(0, 0),
               x_sequence=time, label="input_signal", line_style="--",
               marker='.')

plotter.pre_plot("Pulse Signal")   # stores page 1, resets assignments

# -- Page 2 --
plotter.append_sequence_name(scaled_signal, "scaled_signal")

plotter.assign("scaled_signal", column=0, row=0, position=(0, 0),
               x_sequence=time, label="scaled_signal (x2)", line_style="-",
               marker='')

plotter.pre_plot("Scaled Signal")  # stores page 2, resets assignments

# Launch Dash – both pages appear as tabs in the browser
plotter.plot()
