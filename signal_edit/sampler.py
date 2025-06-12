"""
This module provides functionality for generating and sampling pulse signals,
which are commonly used in signal processing and simulation tasks. The main
class, PulseGenerator, offers static methods to create the time-value points
of a pulse waveform and to sample this waveform at specified intervals.

Classes:
    PulseGenerator:
        Contains static methods to:
            - Generate the time and value points for a pulse signal based on
                configurable parameters such as start time, period, pulse width,
                amplitude, duration, and number of pulses.
            - Sample the generated pulse signal at a specified sampling interval,
                returning the sampled points as a NumPy array.
"""
import numpy as np
from scipy.interpolate import interp1d


class Sampler:
    @staticmethod
    def modify_same_sample_time_value(data, interval):
        """
        Modifies a 2D NumPy array of time-series data to adjust duplicate or closely spaced time values.

        If the first two rows have the same time value, the first row is removed. Then, for each subsequent row,
        if two consecutive time values are the same (starting from the third row), the previous time value is adjusted
        backwards by a calculated difference, but not more than the specified interval.

        Args:
            data (np.ndarray): A 2D NumPy array where the first column represents time values.
            interval (float): The maximum allowed adjustment for duplicate time values.

        Returns:
            np.ndarray: The modified data array with adjusted time values to avoid duplicates.
        """
        if data[0, 0] == data[1, 0]:
            data = data[1:, :]

        for i, val in enumerate(data):
            if i == 0:
                data_out = data[0, :]
            else:
                if (i >= 2) and (data[i, 0] == data[i - 1, 0]):

                    time_dif = data[i, 0] - (data[i - 1, 0] +
                                             data[i - 2, 0]) / 2.0

                    if time_dif >= interval:
                        time_dif = interval

                    data_out[i - 1, 0] = data[i - 1, 0] - time_dif

                data_out = np.vstack((data_out, data[i, :]))

        return data_out

    @staticmethod
    def create_periodical(data, start_time, end_time, sampling_interval):
        """
        Generates periodical samples from input data within a specified time range and sampling interval.

        This function interpolates the input data to ensure consistent sampling intervals, then samples the data
        at regular intervals between `start_time` and `end_time` (inclusive), using linear interpolation.

        Args:
            data (np.ndarray): Input data as a 2D NumPy array with shape (n_samples, 2), where the first column
                represents time points and the second column represents corresponding values.
            start_time (float): The starting time for sampling.
            end_time (float): The ending time for sampling.
            sampling_interval (float): The interval at which to sample the data.

        Returns:
            tuple:
                sample_times (np.ndarray): 2D array of sampled time points with shape (n_samples, 1).
                sampled_values (np.ndarray): 2D array of interpolated values at the sampled time points with shape (n_samples, 1).
        """
        data_inter = Sampler.modify_same_sample_time_value(
            data, sampling_interval)

        time_points = data_inter[:, 0]
        value_points = data_inter[:, 1]

        interp_func = interp1d(time_points, value_points,
                               kind='linear', fill_value='extrapolate')

        sample_times = np.arange(start_time, end_time, sampling_interval)
        sample_times = np.append(sample_times, end_time)

        sampled_values = interp_func(sample_times).reshape(-1, 1)

        sample_times = sample_times.reshape(-1, 1)

        return sample_times, sampled_values


class PulseGenerator:
    @staticmethod
    def generate_pulse_points(start_time, period, pulse_width, pulse_amplitude, duration, number_of_pulse=np.inf):
        """
        Generate pulse points for a pulse signal.

        Parameters:
            start_time (float): The start time of the pulse [s].
            period (float): The period of the pulse [s].
            pulse_width (float): The width of the pulse as a percentage of the period [%].
            pulse_amplitude (float): The amplitude of the pulse.
            duration (float): Total duration of the signal [s].
            number_of_pulse (int): Number of pulses to generate. Default is np.inf for infinite pulses.

        Returns:
            np.ndarray: Array of input points with columns [time, value].
        """
        time_points = []
        value_points = []
        pulse_count = 0

        if start_time > 0.0:
            time_points.append(0.0)
            value_points.append(0.0)

        current_time = start_time
        while current_time <= duration:
            # Pulse ON
            time_points.append(current_time)
            value_points.append(0.0)
            time_points.append(current_time)
            value_points.append(pulse_amplitude)

            pulse_on_duration = period * (pulse_width / 100.0)
            current_time += pulse_on_duration

            if current_time > duration:
                break

            # Pulse OFF
            time_points.append(current_time)
            value_points.append(pulse_amplitude)
            time_points.append(current_time)
            value_points.append(0.0)

            pulse_off_duration = period - pulse_on_duration
            current_time += pulse_off_duration

            pulse_count += 1
            if pulse_count >= number_of_pulse:
                break

        if time_points[-1] < duration:
            time_points.append(duration)
            value_points.append(value_points[-1])

        return np.array(list(zip(time_points, value_points)))

    @staticmethod
    def sample_pulse(sampling_interval, start_time, period, pulse_width, pulse_amplitude, duration, number_of_pulse=np.inf):
        """
        Sample the pulse signal based on the provided parameters.

        Parameters:
            sampling_interval (float): Sampling interval [s].
            start_time (float): The start time of the pulse [s].
            period (float): The period of the pulse [s].
            pulse_width (float): The width of the pulse as a percentage of the period [%].
            pulse_amplitude (float): The amplitude of the pulse.
            duration (float): Total duration of the signal [s].
            number_of_pulse (int): Number of pulses to generate. Default is np.inf for infinite pulses.

        Returns:
            np.ndarray: Array of sampled points with columns [time, value].
        """
        input_points = PulseGenerator.generate_pulse_points(
            start_time, period, pulse_width, pulse_amplitude, duration, number_of_pulse)

        return Sampler.create_periodical(input_points, 0.0, duration, sampling_interval)
