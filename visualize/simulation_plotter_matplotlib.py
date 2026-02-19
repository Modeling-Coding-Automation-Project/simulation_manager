"""
This module provides the SimulationPlotter class,
which facilitates the collection, organization,
and visualization of simulation signal data.
It allows users to append individual signals or
sequences of signals, assign them to specific subplots,
and configure plotting parameters such as labels, line styles, and markers.
The class supports both automatic and manual naming of
signals and x-axis sequences, and manages subplot arrangements
for flexible and customizable visualization of simulation results.

Classes:
    SimulationPlotter:
        A class for managing and visualizing simulation signals.
        It provides methods to append signals, assign them to subplots,
         and generate plots with customizable appearance and layout.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import mplcursors
import numpy as np
import pickle
from datetime import datetime
import inspect

if os.name == 'nt':
    plt.rcParams['font.family'] = 'Meiryo'
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

DUMP_FOLDER_PATH = "./cache/simulation_plotter_dumps/"


class SubplotsInfo:
    def __init__(self, signal_name, shape,
                 column, row, x_sequence,
                 x_sequence_name, line_style,
                 marker, label):

        self.signal_name = signal_name
        self.shape = shape
        self.column = column
        self.row = row

        self.x_sequence = x_sequence
        self.x_sequence_name = x_sequence_name

        self.line_style = line_style
        self.label = label
        self.marker = marker


class Configuration:
    def __init__(self):
        self.subplots_shape = np.zeros((2, 1), dtype=int)
        self.subplots_signals_list = []

        self.window_width_base = 6
        self.window_height_base = 4

        self.window_width_each_subplot = 4
        self.window_height_each_subplot = 2

        self.dual_cursor_mode = False


class MplZoomHelper:
    def __init__(self, ax, zoom_rate=0.9):
        self.ax = ax
        self.ctrl_flag = False
        self.zoom_rate = zoom_rate
        self.canvas = ax.figure.canvas

        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_key_press(self, event):
        """
        Handles key press events to track Control key state.

        Sets the ctrl_flag to True when the Control key is pressed,
        enabling vertical zoom mode for scroll events.

        Args:
            event: The key press event object containing key information.
        """
        if event.key == 'control':
            self.ctrl_flag = True

    def on_key_release(self, event):
        """
        Handles key release events to track Control key state.

        Sets the ctrl_flag to False when the Control key is released,
        disabling vertical zoom mode for scroll events.

        Args:
            event: The key release event object containing key information.
        """
        if event.key == 'control':
            self.ctrl_flag = False

    def on_scroll(self, event):
        """
        Handles scroll events to perform zooming on the plot.
        Zooms in or out on the x-axis or y-axis depending on
         the state of the ctrl_flag (Control key).
        Args:
            event: The scroll event object containing scroll information.
        """
        if event.inaxes is not self.ax:
            return

        if event.xdata is None and event.ydata is None:
            return

        if self.ctrl_flag:
            self._on_scroll_y(event)
        else:
            self._on_scroll_x(event)

        self.ax.figure.canvas.draw_idle()

    def _on_scroll_x(self, event):
        """
        Performs zooming on the x-axis based on scroll events.
        Args:
            event: The scroll event object containing scroll information.
        """
        x_pos = event.xdata
        if x_pos is None:
            return
        min_value, max_value = self.ax.get_xlim()
        if event.button == 'up':
            new_min = x_pos - (x_pos - min_value) * self.zoom_rate
            new_max = (max_value - x_pos) * self.zoom_rate + x_pos
        elif event.button == 'down':
            new_min = x_pos - (x_pos - min_value) / self.zoom_rate
            new_max = (max_value - x_pos) / self.zoom_rate + x_pos
        else:
            return
        self.ax.set_xlim(new_min, new_max)

    def _on_scroll_y(self, event):
        """
        Performs zooming on the y-axis based on scroll events.
        Args:
            event: The scroll event object containing scroll information.
        """
        y_pos = event.ydata
        if y_pos is None:
            return
        min_value, max_value = self.ax.get_ylim()
        if event.button == 'up':
            new_min = y_pos - (y_pos - min_value) * self.zoom_rate
            new_max = (max_value - y_pos) * self.zoom_rate + y_pos
        elif event.button == 'down':
            new_min = y_pos - (y_pos - min_value) / self.zoom_rate
            new_max = (max_value - y_pos) / self.zoom_rate + y_pos
        else:
            return
        self.ax.set_ylim(new_min, new_max)


class SimulationPlotterMatplotlib:

    def __init__(self, activate_dump=False):
        self.configuration = Configuration()
        self.name_to_object_dictionary = {}

        self.activate_dump = activate_dump

        self.subplot_cursors = {}

    def append(self, signal_object):
        """
        Appends a signal object to the internal name-to-object dictionary
         using the variable name from the caller's local scope as the key.

        This method inspects the caller's local variables to determine the
         variable name referencing the provided signal_object.
        If the variable name already exists in the dictionary,
         the signal_object is appended to the existing list.
        Otherwise, a new list is created for that variable name.

        Args:
            signal_object: The signal object to be appended and tracked.

        Raises:
            ValueError: If the variable name for signal_object
             cannot be determined from the caller's local scope.
        """
        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the object_name that matches the matrix_in value
        object_name = None
        for name, value in caller_locals.items():
            if value is signal_object:
                object_name = name
                break

        # %% append object
        signal_copy = np.copy(signal_object)
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_copy)
        else:
            self.name_to_object_dictionary[object_name] = [signal_copy]

    def append_name(self, signal_object, object_name):
        """
        Appends a signal object to the list associated with the given object name in the name_to_object_dictionary.

        If the object name already exists as a key in the dictionary, the signal object is appended to the existing list.
        If the object name does not exist, a new list is created with the signal object as its first element.

        Args:
            signal_object: The signal object to be associated with the object name.
            object_name (str): The key representing the name to which the signal object should be appended.
        """
        signal_copy = np.copy(signal_object)
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_copy)
        else:
            self.name_to_object_dictionary[object_name] = [signal_copy]

    def append_sequence(self, signal_sequence_object):
        """
        Appends a sequence of signal objects to the internal name-to-object dictionary.

        This method inspects the caller's local variables to determine the variable name
        of the provided signal_sequence_object. It then appends each element of the sequence
        (after reshaping to a column vector) to the list associated with that variable name
        in self.name_to_object_dictionary. If the variable name does not exist in the dictionary,
        a new entry is created.

        Args:
            signal_sequence_object (iterable): An iterable of signal objects (e.g., numpy arrays)
                to be appended. Each element is reshaped to a column vector before appending.

        Side Effects:
            Modifies self.name_to_object_dictionary by appending or creating entries for the
            provided signal sequence.

        Note:
            The method relies on inspecting the caller's frame to determine the variable name
            of the signal_sequence_object. This may not work as expected in all contexts,
            such as when the object is passed as an expression or temporary value.
        """
        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the object_name that matches the matrix_in value
        object_name = None
        for name, value in caller_locals.items():
            if value is signal_sequence_object:
                object_name = name
                break

        # %% append object
        for i in range(len(signal_sequence_object)):
            signal_copy = np.copy(signal_sequence_object[i])
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_copy.reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_copy.reshape(-1, 1)]

    def append_sequence_name(self, signal_sequence_object, object_name):
        """
        Appends reshaped elements from a signal sequence to a dictionary entry keyed by object_name.

        If the object_name already exists in the name_to_object_dictionary, each element of the
        signal_sequence_object is reshaped to a column vector and appended to the existing list.
        If the object_name does not exist, a new list is created with the reshaped elements.

        Args:
            signal_sequence_object (iterable): An iterable of signal data (e.g., numpy arrays)
             to be reshaped and stored.
            object_name (str): The key under which the reshaped signal data
             will be stored in the dictionary.

        Side Effects:
            Modifies self.name_to_object_dictionary by appending or creating entries for object_name.
        """
        for i in range(len(signal_sequence_object)):
            signal_copy = np.copy(signal_sequence_object[i])
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_copy.reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_copy.reshape(-1, 1)]

    def assign(self, signal_name, position,
               column=0, row=0, x_sequence=None, x_sequence_name=None,
               line_style="-", marker="", label=""):
        """
        Assigns a signal to a subplot configuration for visualization.

        Parameters:
            signal_name (str): The name of the signal to assign.
            position (tuple or list): The (x, y) position of the subplot.
            column (int, optional): The column index for subplot placement. Defaults to 0.
            row (int, optional): The row index for subplot placement. Defaults to 0.
            x_sequence (array-like or str, optional): The x-axis data sequence or its name.
             If a string, it is resolved via `name_to_object_dictionary`.
            x_sequence_name (str, optional): The name of the x_sequence.
             If not provided and x_sequence is not None, attempts to infer the
              variable name from the caller's scope.
            line_style (str, optional): The line style for plotting. Defaults to "-".
            marker (str, optional): The marker style for plotting. Defaults to "".
            label (str, optional): The label for the plot legend. Defaults to "".

        Side Effects:
            Appends a SubplotsInfo object to
            `self.configuration.subplots_signals_list` with the provided configuration.

        Notes:
            If `x_sequence_name` is not provided and `x_sequence` is given,
             the method attempts to infer the variable name of
             `x_sequence` from the caller's local variables.
            If `x_sequence` is a string, it is used as a key to retrieve
            the actual sequence object from `self.name_to_object_dictionary`.
        """
        this_x_sequence_name = ""
        if (x_sequence is not None) and (x_sequence_name is None):
            # %% inspect arguments
            # Get the caller's frame
            frame = inspect.currentframe().f_back
            # Get the caller's local variables
            caller_locals = frame.f_locals
            # Find the object_name that matches the matrix_in value
            this_x_sequence_name = None
            for name, value in caller_locals.items():
                if value is x_sequence:
                    this_x_sequence_name = name
                    break
        else:
            this_x_sequence_name = x_sequence_name

        if (x_sequence is not None) and isinstance(x_sequence, str):
            this_x_sequence_name = x_sequence
            x_sequence = self.name_to_object_dictionary[x_sequence]

        # %% assign object
        shape = np.array([[position[0]], [position[1]]], dtype=int)

        self.configuration.subplots_signals_list.append(
            SubplotsInfo(signal_name, shape,
                         column, row, x_sequence,
                         this_x_sequence_name, line_style,
                         marker, label))

    def assign_all(self, signal_name, position,
                   x_sequence=None, x_sequence_name=None,
                   line_style="-", marker="", label=""):
        """
        Assigns all elements of a signal (by name) to be plotted, iterating over its columns and rows.

        Parameters:
            signal_name (str): The name of the signal to assign for plotting.
            position (Any): The position or subplot index where the signal should be plotted.
            x_sequence (array-like, optional): The x-axis data sequence for the plot.
             If not provided, defaults to None.
            x_sequence_name (str, optional): The name of the x_sequence variable.
            If not provided, attempts to infer from caller's local variables.
            line_style (str, optional): The line style for the plot (default is "-").
            marker (str, optional): The marker style for the plot (default is "").
            label (str, optional): The base label for the plot. If not provided, uses signal_name.

        Notes:
            - For each element in the signal's matrix (by column and row),
            this method calls `self.assign` to register the plot.
            - If x_sequence_name is not provided, it tries to infer the variable
             name from the caller's local scope.
            - The label for each plot is constructed as "{label}_{i}_{j}"
             where i and j are the column and row indices.
        """
        this_x_sequence_name = ""
        if (x_sequence is not None) and (x_sequence_name is None):
            # %% inspect arguments
            # Get the caller's frame
            frame = inspect.currentframe().f_back
            # Get the caller's local variables
            caller_locals = frame.f_locals
            # Find the object_name that matches the matrix_in value
            this_x_sequence_name = None
            for name, value in caller_locals.items():
                if value is x_sequence:
                    this_x_sequence_name = name
                    break
        else:
            this_x_sequence_name = x_sequence_name

        col_size = self.name_to_object_dictionary[signal_name][0].shape[0]
        row_size = self.name_to_object_dictionary[signal_name][0].shape[1]

        if label == "":
            label = signal_name

        for i in range(col_size):
            for j in range(row_size):
                label_text = label + "_" + str(i) + "_" + str(j)
                self.assign(signal_name, position=position,
                            column=i, row=j,
                            x_sequence=x_sequence,
                            x_sequence_name=this_x_sequence_name,
                            line_style=line_style, marker=marker,
                            label=label_text)

    def _dump_simulation_plotter(self, filename=None):
        """
        Internal helper to dump the SimulationPlotter instance (or a
        sanitized snapshot) into a timestamped .npz file.

        If direct pickling of `self` fails, a sanitized `snapshot`
        dictionary is constructed where non-pickleable members are
        replaced with `None` (except for `subplot_cursors`, which
        records only `x_data`/`y_data`).
        """
        try:
            pickled = pickle.dumps(self)
        except Exception:
            snapshot = {}
            for k, v in self.__dict__.items():
                if k == 'subplot_cursors':
                    sc = {}
                    for sk, sinfo in v.items():
                        sc[str(sk)] = {
                            'x_data': sinfo.get('x_data'),
                            'y_data': sinfo.get('y_data')
                        }
                    snapshot[k] = sc
                else:
                    try:
                        pickle.dumps(v)
                        snapshot[k] = v
                    except Exception:
                        snapshot[k] = None

            pickled = pickle.dumps(snapshot)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        if filename is None:
            filename = f"SimulationPlotterData_{timestamp}.npz"

        save_file_path = os.path.join(DUMP_FOLDER_PATH, filename)
        os.makedirs(DUMP_FOLDER_PATH, exist_ok=True)

        try:
            np.savez(save_file_path, simulation_plotter=pickled)
        except Exception as e:
            print(f"Failed to save SimulationPlotter dump: {e}")

    def pre_plot(self, suptitle=""):
        """
        Prepares and displays subplots for visualizing simulation signals based on the current configuration.

        This method determines the layout and size of the figure, creates subplots, and plots each signal
        specified in the configuration's `subplots_signals_list`. It handles both single and multiple subplot
        arrangements, supports custom x-axis sequences, and applies labels, line styles, and markers as specified
        in each signal's configuration. Interactive cursors are enabled for each subplot with dual-cursor mode support.

        Args:
            suptitle (str, optional): The title for the entire figure. Defaults to an empty string.

        Returns:
            None

        Side Effects:
            - Displays the generated figure with the plotted signals.
            - Updates `self.configuration.subplots_shape` with the determined subplot grid shape.
            - Prints a message if there are no subplots to show.
        """

        subplots_signals_list = self.configuration.subplots_signals_list

        if len(subplots_signals_list) == 0:
            print("No subplots to show.")
            return

        if self.activate_dump:
            self._dump_simulation_plotter()

        shape = np.zeros((2, 1), dtype=int)
        for signal_info in subplots_signals_list:
            if shape[0, 0] < signal_info.shape[0, 0] + 1:
                shape[0, 0] = signal_info.shape[0, 0] + 1
            if shape[1, 0] < signal_info.shape[1, 0] + 1:
                shape[1, 0] = signal_info.shape[1, 0] + 1

        self.configuration.subplots_shape = shape

        figure_size = (self.configuration.window_width_base +
                       self.configuration.window_width_each_subplot *
                       (shape[1, 0] - 1) + 2,
                       self.configuration.window_height_base +
                       self.configuration.window_height_each_subplot * (shape[0, 0] - 1))

        fig, axs = plt.subplots(shape[0, 0], shape[1, 0], figsize=figure_size)
        plt.subplots_adjust(right=0.85)
        fig.suptitle(suptitle)

        self.subplot_cursors = {}

        for signal_info in subplots_signals_list:
            signal_object_list = self.name_to_object_dictionary[signal_info.signal_name]

            steps = len(signal_object_list)

            # Build x-axis sequence as a (steps,1) numeric array.
            # Normalize various possible shapes (e.g., (N,1), (N,), list of scalars)
            if signal_info.x_sequence is not None:
                x_arr = np.asarray(signal_info.x_sequence)
                # Flatten any (N,1) or higher-dim arrays to 1D
                x_arr = x_arr.reshape(-1)
                # If provided sequence is shorter than steps, pad with last value
                if x_arr.shape[0] < steps:
                    if x_arr.shape[0] == 0:
                        x_arr = np.zeros(steps)
                    else:
                        pad = np.empty(steps)
                        pad[:] = x_arr[-1]
                        pad[: x_arr.shape[0]] = x_arr
                        x_arr = pad
                x_sequence_signal = np.asarray(x_arr[:steps]).reshape(steps, 1)
            else:
                x_sequence_signal = np.arange(steps).reshape(steps, 1)

            signal = np.zeros((steps, 1))
            if isinstance(signal_object_list[0], np.ndarray):
                for i in range(steps):
                    signal[i, 0] = signal_object_list[i][signal_info.column,
                                                         signal_info.row]
            else:
                for i in range(steps):
                    signal[i, 0] = signal_object_list[i]

            if signal_info.label == "":
                label_name = signal_info.signal_name + \
                    f"[{signal_info.column}, {signal_info.row}]"
            else:
                label_name = signal_info.label

            if shape[0] == 1 and shape[1] == 1:
                ax = axs
            elif shape[0] == 1:
                ax = axs[signal_info.shape[1, 0]]
            elif shape[1] == 1:
                ax = axs[signal_info.shape[0, 0]]
            else:
                ax = axs[signal_info.shape[0, 0], signal_info.shape[1, 0]]

            line, = ax.plot(x_sequence_signal, signal,
                            label=label_name,
                            linestyle=signal_info.line_style, marker=signal_info.marker)

            subplot_key = (signal_info.shape[0, 0], signal_info.shape[1, 0])
            if subplot_key not in self.subplot_cursors:
                # cursor_mpl = mplcursors.cursor(ax, multiple=True)
                cursor_mpl = mplcursors.cursor(ax, multiple=False)
                cursor_mpl.enabled = True
                # instantiate zoom helper for this axis
                zoom_helper = MplZoomHelper(ax)
                self.subplot_cursors[subplot_key] = {
                    'ax': ax,
                    'cursor_mpl': cursor_mpl,
                    'cursor1': None,
                    'cursor2': None,
                    'text1': None,
                    'text2': None,
                    'text_diff': None,
                    'x_data': x_sequence_signal.flatten(),
                    'y_data': signal.flatten(),
                    'zoom_helper': zoom_helper
                }
            else:
                cursor_info = self.subplot_cursors[subplot_key]
                mplcursors.cursor(line, multiple=False)

            ax.legend()
            ax.set_xlabel(signal_info.x_sequence_name)
            ax.grid(True)

        check_ax = plt.axes([0.0, 0.0, 0.05, 0.05])
        check = CheckButtons(check_ax, ["Dual cursor\nmode"], [False])

        def clear_dual_cursors(subplot_key):
            """
            Deletes dual cursors and associated texts from the specified subplot.
            """
            cursor_info = self.subplot_cursors[subplot_key]
            if cursor_info['cursor1'] is not None:
                cursor_info['cursor1'].remove()
                cursor_info['cursor1'] = None
            if cursor_info['cursor2'] is not None:
                cursor_info['cursor2'].remove()
                cursor_info['cursor2'] = None
            if cursor_info['text1'] is not None:
                cursor_info['text1'].remove()
                cursor_info['text1'] = None
            if cursor_info['text2'] is not None:
                cursor_info['text2'].remove()
                cursor_info['text2'] = None
            if cursor_info['text_diff'] is not None:
                cursor_info['text_diff'].remove()
                cursor_info['text_diff'] = None

        def clear_mplcursors(subplot_key):
            """
            Deletes all mplcursor selections from the specified subplot.
            """
            cursor_info = self.subplot_cursors[subplot_key]
            for sel in list(cursor_info['cursor_mpl'].selections):
                cursor_info['cursor_mpl'].remove_selection(sel)

        def toggle_mode(label):
            """
            Handles the event when the checkbox is clicked.
            """
            self.configuration.dual_cursor_mode = check.get_status()[0]

            for subplot_key, cursor_info in self.subplot_cursors.items():
                if self.configuration.dual_cursor_mode:

                    cursor_info['cursor_mpl'].enabled = False
                    clear_mplcursors(subplot_key)
                else:
                    cursor_info['cursor_mpl'].enabled = True
                    clear_dual_cursors(subplot_key)

            fig.canvas.draw_idle()

        def on_click(event):
            """
            Handles the click event.
            """
            if not self.configuration.dual_cursor_mode:
                return

            clicked_subplot_key = None
            for subplot_key, cursor_info in self.subplot_cursors.items():
                if event.inaxes == cursor_info['ax']:
                    clicked_subplot_key = subplot_key
                    break

            if clicked_subplot_key is None:
                return

            cursor_info = self.subplot_cursors[clicked_subplot_key]
            ax = cursor_info['ax']
            x_data = cursor_info['x_data']
            y_data = cursor_info['y_data']

            cx = event.xdata
            # Find the index of the x value closest to the click position
            idx = np.abs(x_data - cx).argmin()
            cy = y_data[idx]

            # Left click -> Cursor 1
            if event.button == 1:
                if cursor_info['cursor1'] is not None:
                    cursor_info['cursor1'].remove()
                if cursor_info['text1'] is not None:
                    cursor_info['text1'].remove()

                cursor_info['cursor1'] = ax.axvline(
                    cx, color="red", linestyle="--", linewidth=1.2)
                cursor_info['text1'] = ax.text(
                    0.02, 0.95,
                    f"Cursor1: x={cx:.3f}, y={cy:.3f}",
                    transform=ax.transAxes,
                    color="red",
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            # Right click -> Cursor 2
            elif event.button == 3:
                if cursor_info['cursor2'] is not None:
                    cursor_info['cursor2'].remove()
                if cursor_info['text2'] is not None:
                    cursor_info['text2'].remove()

                cursor_info['cursor2'] = ax.axvline(
                    cx, color="blue", linestyle="--", linewidth=1.2)
                cursor_info['text2'] = ax.text(
                    0.02, 0.85,
                    f"Cursor2: x={cx:.3f}, y={cy:.3f}",
                    transform=ax.transAxes,
                    color="blue",
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            if cursor_info['cursor1'] is not None and cursor_info['cursor2'] is not None:
                x1 = cursor_info['cursor1'].get_xdata()[0]
                x2 = cursor_info['cursor2'].get_xdata()[0]
                diff = abs(x2 - x1)

                if cursor_info['text_diff'] is not None:
                    cursor_info['text_diff'].remove()

                cursor_info['text_diff'] = ax.text(
                    0.02, 0.75,
                    f"Δx = {diff:.3f}",
                    transform=ax.transAxes,
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
                )

            fig.canvas.draw_idle()

        check.on_clicked(toggle_mode)
        fig.canvas.mpl_connect("button_press_event", on_click)

    def plot(
        self,
        suptitle="",
        dump_file_path=None
    ):
        """
        Plots the simulation data either from the current instance
          or loads from a dump file.

        Args:
            suptitle (str, optional): The title for the entire figure.
            Defaults to an empty string.
            dump_file_path (str, optional): Path to a dump file or
              directory containing dump files.
            If None, plots the current instance. Defaults to None.
            If dump_file_path is None, the method calls pre_plot()
            on the current instance
            to generate and display the plots. If a dump_file_path is provided,
            it attempts to load the dump file,
            reconstruct the SimulationPlotter instance,
            and calls its plot() method.
        """

        if dump_file_path is None:
            self.pre_plot(suptitle)
            plt.show()
            return

        path = dump_file_path
        if os.path.isdir(path):
            npz_files = [os.path.join(path, f) for f in os.listdir(
                path) if f.lower().endswith('.npz')]
            if not npz_files:
                print(f"No .npz files found in directory: {path}")
                return
            path = max(npz_files, key=os.path.getmtime)

        if not os.path.exists(path):
            alt = os.path.join(DUMP_FOLDER_PATH, path)
            if os.path.exists(alt):
                path = alt

        try:
            with np.load(path, allow_pickle=True) as npz:
                pickled = npz['simulation_plotter']
                if isinstance(pickled, np.ndarray):
                    pickled = pickled.item()
            loaded = pickle.loads(pickled)
        except Exception as e:
            print(f"Failed to load dump file '{path}': {e}")
            return

        if hasattr(loaded, 'plot') and callable(getattr(loaded, 'plot')):
            try:
                loaded.activate_dump = False
                loaded.plot(suptitle)
            except Exception as e:
                print(f"Failed to call plot() on loaded object: {e}")
            return

        if isinstance(loaded, dict):
            sp = SimulationPlotter(activate_dump=False)
            for k, v in loaded.items():
                try:
                    setattr(sp, k, v)
                except Exception:
                    pass
            try:
                sp.plot(suptitle)
            except Exception as e:
                print(f"Failed to plot reconstructed SimulationPlotter: {e}")
            return

        print("Loaded dump does not contain a usable SimulationPlotter object.")
