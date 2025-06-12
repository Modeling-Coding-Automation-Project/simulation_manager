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
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import inspect


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


class SimulationPlotter:

    def __init__(self):
        self.configuration = Configuration()
        self.name_to_object_dictionary = {}

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
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_object)
        else:
            self.name_to_object_dictionary[object_name] = [signal_object]

    def append_name(self, signal_object, object_name):
        """
        Appends a signal object to the list associated with the given object name in the name_to_object_dictionary.

        If the object name already exists as a key in the dictionary, the signal object is appended to the existing list.
        If the object name does not exist, a new list is created with the signal object as its first element.

        Args:
            signal_object: The signal object to be associated with the object name.
            object_name (str): The key representing the name to which the signal object should be appended.
        """
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_object)
        else:
            self.name_to_object_dictionary[object_name] = [signal_object]

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
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_sequence_object[i].reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_sequence_object[i].reshape(-1, 1)]

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
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_sequence_object[i].reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_sequence_object[i].reshape(-1, 1)]

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

        col_size = eval(
            f"self.name_to_object_dictionary[\"{signal_name}\"][0].shape[0]")
        row_size = eval(
            f"self.name_to_object_dictionary[\"{signal_name}\"][0].shape[1]")

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

    def pre_plot(self, suptitle=""):
        """
        Prepares and displays subplots for visualizing simulation signals based on the current configuration.

        This method determines the layout and size of the figure, creates subplots, and plots each signal
        specified in the configuration's `subplots_signals_list`. It handles both single and multiple subplot
        arrangements, supports custom x-axis sequences, and applies labels, line styles, and markers as specified
        in each signal's configuration. Interactive cursors are enabled for each subplot.

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

        shape = np.zeros((2, 1), dtype=int)
        for signal_info in subplots_signals_list:
            if shape[0, 0] < signal_info.shape[0, 0] + 1:
                shape[0, 0] = signal_info.shape[0, 0] + 1
            if shape[1, 0] < signal_info.shape[1, 0] + 1:
                shape[1, 0] = signal_info.shape[1, 0] + 1

        self.configuration.subplots_shape = shape

        figure_size = (self.configuration.window_width_base +
                       self.configuration.window_width_each_subplot *
                       (shape[1, 0] - 1),
                       self.configuration.window_height_base +
                       self.configuration.window_height_each_subplot * (shape[0, 0] - 1))

        fig, axs = plt.subplots(shape[0, 0], shape[1, 0], figsize=figure_size)
        fig.suptitle(suptitle)

        for signal_info in subplots_signals_list:
            signal_object_list = self.name_to_object_dictionary[signal_info.signal_name]

            steps = len(signal_object_list)

            x_sequence_signal = np.zeros((steps, 1))
            if signal_info.x_sequence is not None:
                for i in range(steps):
                    x_sequence_signal[i, 0] = signal_info.x_sequence[i]
            else:
                x_sequence_signal = np.arange(steps)

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

            ax.plot(x_sequence_signal, signal,
                    label=label_name,
                    linestyle=signal_info.line_style, marker=signal_info.marker)
            mplcursors.cursor(ax)
            ax.legend()
            ax.set_xlabel(signal_info.x_sequence_name)
            ax.grid(True)

    def plot(self, suptitle=""):
        """
        Generates and displays the simulation plot.

        This method prepares the plot by calling `pre_plot` with an optional super title,
        then displays the plot window using matplotlib's `plt.show()`.

        Args:
            suptitle (str, optional): The super title for the plot. Defaults to an empty string.
        """
        self.pre_plot(suptitle)

        plt.show()
