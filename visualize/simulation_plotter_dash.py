"""
This module provides the SimulationPlotterDash class,
which facilitates the collection, organization,
and visualization of simulation signal data
using Plotly and Dash for interactive web-based visualization.

It implements the same interface as SimulationPlotter
but replaces matplotlib with Plotly for charts and Dash for the UI,
enabling interactive features such as zoom, pan,
hover tooltips, and dual cursor mode in a web browser.

Classes:
    SimulationPlotterDash:
        A class for managing and visualizing simulation signals
        using Plotly and Dash.
        It provides methods to append signals, assign them to subplots,
        and generate plots with customizable appearance and layout.
"""
import os
import pickle
import inspect
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context

DUMP_FOLDER_PATH = "./cache/simulation_plotter_dumps/"

_LINE_STYLE_MAP = {
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
}

_MARKER_MAP = {
    ".": "circle",
    "o": "circle",
    "^": "triangle-up",
    "v": "triangle-down",
    "s": "square",
    "*": "star",
    "+": "cross",
    "x": "x",
    "D": "diamond",
    "d": "diamond-thin",
    "p": "pentagon",
    "h": "hexagon",
}


def _convert_line_style(mpl_style):
    """Convert a matplotlib line style string to a Plotly dash string."""
    return _LINE_STYLE_MAP.get(mpl_style, "solid")


def _convert_marker(mpl_marker):
    """Convert a matplotlib marker string to a Plotly marker symbol string."""
    return _MARKER_MAP.get(mpl_marker, "circle")


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
        self.dual_cursor_mode = False


class SimulationPlotterDash:

    def __init__(self, activate_dump=False):
        self.configuration = Configuration()
        self.name_to_object_dictionary = {}
        self.activate_dump = activate_dump

    def append(self, signal_object):
        """
        Appends a signal object to the internal name-to-object dictionary
         using the variable name from the caller's local scope as the key.

        Args:
            signal_object: The signal object to be appended and tracked.

        Raises:
            ValueError: If the variable name for signal_object
             cannot be determined from the caller's local scope.
        """
        frame = inspect.currentframe().f_back
        caller_locals = frame.f_locals
        object_name = None
        for name, value in caller_locals.items():
            if value is signal_object:
                object_name = name
                break
        del frame

        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_object)
        else:
            self.name_to_object_dictionary[object_name] = [signal_object]

    def append_name(self, signal_object, object_name):
        """
        Appends a signal object to the list associated with the given
        object name in the name_to_object_dictionary.

        Args:
            signal_object: The signal object to be associated with the object name.
            object_name (str): The key representing the name to which
             the signal object should be appended.
        """
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_object)
        else:
            self.name_to_object_dictionary[object_name] = [signal_object]

    def append_sequence(self, signal_sequence_object):
        """
        Appends a sequence of signal objects to the internal
        name-to-object dictionary.

        Args:
            signal_sequence_object (iterable): An iterable of signal objects
             (e.g., numpy arrays) to be appended. Each element is reshaped
             to a column vector before appending.
        """
        frame = inspect.currentframe().f_back
        caller_locals = frame.f_locals
        object_name = None
        for name, value in caller_locals.items():
            if value is signal_sequence_object:
                object_name = name
                break
        del frame

        for i in range(len(signal_sequence_object)):
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_sequence_object[i].reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_sequence_object[i].reshape(-1, 1)]

    def append_sequence_name(self, signal_sequence_object, object_name):
        """
        Appends reshaped elements from a signal sequence to a dictionary
        entry keyed by object_name.

        Args:
            signal_sequence_object (iterable): An iterable of signal data
             (e.g., numpy arrays) to be reshaped and stored.
            object_name (str): The key under which the reshaped signal data
             will be stored in the dictionary.
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
            column (int, optional): The column index for subplot placement.
             Defaults to 0.
            row (int, optional): The row index for subplot placement.
             Defaults to 0.
            x_sequence (array-like or str, optional): The x-axis data sequence
             or its name.
            x_sequence_name (str, optional): The name of the x_sequence.
            line_style (str, optional): The line style for plotting.
             Defaults to "-".
            marker (str, optional): The marker style for plotting.
             Defaults to "".
            label (str, optional): The label for the plot legend.
             Defaults to "".
        """
        this_x_sequence_name = ""
        if (x_sequence is not None) and (x_sequence_name is None):
            frame = inspect.currentframe().f_back
            caller_locals = frame.f_locals
            this_x_sequence_name = None
            for name, value in caller_locals.items():
                if value is x_sequence:
                    this_x_sequence_name = name
                    break
            del frame
        else:
            this_x_sequence_name = x_sequence_name

        if (x_sequence is not None) and isinstance(x_sequence, str):
            this_x_sequence_name = x_sequence
            x_sequence = self.name_to_object_dictionary[x_sequence]

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
        Assigns all elements of a signal (by name) to be plotted,
        iterating over its columns and rows.

        Parameters:
            signal_name (str): The name of the signal to assign for plotting.
            position (Any): The position or subplot index where the signal
             should be plotted.
            x_sequence (array-like, optional): The x-axis data sequence
             for the plot.
            x_sequence_name (str, optional): The name of the x_sequence
             variable.
            line_style (str, optional): The line style for the plot
             (default is "-").
            marker (str, optional): The marker style for the plot
             (default is "").
            label (str, optional): The base label for the plot.
             If not provided, uses signal_name.
        """
        this_x_sequence_name = ""
        if (x_sequence is not None) and (x_sequence_name is None):
            frame = inspect.currentframe().f_back
            caller_locals = frame.f_locals
            this_x_sequence_name = None
            for name, value in caller_locals.items():
                if value is x_sequence:
                    this_x_sequence_name = name
                    break
            del frame
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
        Internal helper to dump the SimulationPlotterDash instance into
        a timestamped .npz file.
        """
        try:
            pickled = pickle.dumps(self)
        except Exception:
            snapshot = {}
            for k, v in self.__dict__.items():
                try:
                    pickle.dumps(v)
                    snapshot[k] = v
                except Exception:
                    snapshot[k] = None

            pickled = pickle.dumps(snapshot)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        if filename is None:
            filename = f"SimulationPlotterDashData_{timestamp}.npz"

        save_file_path = os.path.join(DUMP_FOLDER_PATH, filename)
        os.makedirs(DUMP_FOLDER_PATH, exist_ok=True)

        try:
            np.savez(save_file_path, simulation_plotter=pickled)
        except Exception as e:
            print(f"Failed to save SimulationPlotterDash dump: {e}")

    def _build_figure(self, suptitle=""):
        """
        Build the Plotly figure from the current configuration.

        Args:
            suptitle (str): The title for the entire figure.

        Returns:
            tuple: (fig, shape) where fig is the Plotly Figure object
             and shape is a (2,1) numpy array with [n_rows, n_cols].
             Returns (None, None) if no subplots are configured.
        """
        subplots_signals_list = self.configuration.subplots_signals_list

        if len(subplots_signals_list) == 0:
            print("No subplots to show.")
            return None, None

        shape = np.zeros((2, 1), dtype=int)
        for signal_info in subplots_signals_list:
            if shape[0, 0] < signal_info.shape[0, 0] + 1:
                shape[0, 0] = signal_info.shape[0, 0] + 1
            if shape[1, 0] < signal_info.shape[1, 0] + 1:
                shape[1, 0] = signal_info.shape[1, 0] + 1

        self.configuration.subplots_shape = shape

        n_rows = int(shape[0, 0])
        n_cols = int(shape[1, 0])

        subplot_titles = [""] * (n_rows * n_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=False,
        )

        for signal_info in subplots_signals_list:
            signal_object_list = self.name_to_object_dictionary[
                signal_info.signal_name]
            steps = len(signal_object_list)

            if signal_info.x_sequence is not None:
                x_arr = np.asarray(signal_info.x_sequence).reshape(-1)
                if x_arr.shape[0] < steps:
                    if x_arr.shape[0] == 0:
                        x_arr = np.zeros(steps)
                    else:
                        pad = np.empty(steps)
                        pad[:] = x_arr[-1]
                        pad[: x_arr.shape[0]] = x_arr
                        x_arr = pad
                x_sequence_signal = x_arr[:steps]
            else:
                x_sequence_signal = np.arange(steps, dtype=float)

            signal = np.zeros(steps)
            if isinstance(signal_object_list[0], np.ndarray):
                for i in range(steps):
                    signal[i] = signal_object_list[i][signal_info.column,
                                                      signal_info.row]
            else:
                for i in range(steps):
                    signal[i] = signal_object_list[i]

            if signal_info.label == "":
                label_name = (signal_info.signal_name
                              + f"[{signal_info.column}, {signal_info.row}]")
            else:
                label_name = signal_info.label

            plot_row = int(signal_info.shape[0, 0]) + 1
            plot_col = int(signal_info.shape[1, 0]) + 1

            dash_style = _convert_line_style(signal_info.line_style)
            mode = "lines+markers" if signal_info.marker else "lines"
            marker_symbol = (_convert_marker(signal_info.marker)
                             if signal_info.marker else "circle")

            trace = go.Scatter(
                x=x_sequence_signal.tolist(),
                y=signal.tolist(),
                name=label_name,
                mode=mode,
                line=dict(dash=dash_style),
                marker=dict(symbol=marker_symbol),
            )

            fig.add_trace(trace, row=plot_row, col=plot_col)

            fig.update_xaxes(
                title_text=signal_info.x_sequence_name or "",
                showgrid=True,
                row=plot_row,
                col=plot_col,
            )
            fig.update_yaxes(
                showgrid=True,
                row=plot_row,
                col=plot_col,
            )

        fig.update_layout(
            title_text=suptitle,
            height=max(400, 300 * n_rows),
            showlegend=True,
        )

        return fig, shape

    def _run_dash_app(self, fig, shape, port=8050, debug=False):
        """
        Launch the Dash application with the given Plotly figure.

        Args:
            fig: The Plotly Figure object to display.
            shape: A (2,1) numpy array with [n_rows, n_cols].
            port (int): Port number for the Dash server. Defaults to 8050.
            debug (bool): Run Dash in debug mode. Defaults to False.
        """
        app = Dash(__name__)

        graph_height = max(400, 300 * int(shape[0, 0]))

        app.layout = html.Div([
            html.Div(
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '20px',
                    'padding': '8px 12px',
                    'backgroundColor': '#f5f5f5',
                    'borderBottom': '1px solid #ddd',
                },
                children=[
                    dcc.Checklist(
                        id='dual-cursor-toggle',
                        options=[{'label': ' Dual cursor mode',
                                  'value': 'on'}],
                        value=[],
                        style={'fontSize': '14px'},
                    ),
                    html.Div(
                        id='cursor-controls',
                        children=[
                            html.Span('Select cursor: ',
                                      style={'fontSize': '13px'}),
                            dcc.RadioItems(
                                id='cursor-select',
                                options=[
                                    {'label': ' Cursor 1 (red)',
                                     'value': '1'},
                                    {'label': ' Cursor 2 (blue)',
                                     'value': '2'},
                                ],
                                value='1',
                                inline=True,
                                style={'fontSize': '13px'},
                            ),
                        ],
                        style={'display': 'none'},
                    ),
                ],
            ),
            html.Div(
                id='cursor-info',
                style={
                    'fontFamily': 'monospace',
                    'fontSize': '13px',
                    'padding': '6px 12px',
                    'minHeight': '20px',
                    'backgroundColor': '#fffbe6',
                    'borderBottom': '1px solid #ddd',
                    'whiteSpace': 'pre',
                },
            ),
            dcc.Graph(
                id='main-graph',
                figure=fig,
                style={'height': f'{graph_height}px'},
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
                },
            ),
            dcc.Store(
                id='cursor-store',
                data={'1': {}, '2': {}},
            ),
        ])

        @app.callback(
            Output('cursor-controls', 'style'),
            Input('dual-cursor-toggle', 'value'),
        )
        def toggle_cursor_controls(dual_mode_value):
            if dual_mode_value and 'on' in dual_mode_value:
                return {'display': 'flex', 'alignItems': 'center', 'gap': '8px'}
            return {'display': 'none'}

        @app.callback(
            Output('main-graph', 'figure'),
            Output('cursor-info', 'children'),
            Output('cursor-store', 'data'),
            Input('main-graph', 'clickData'),
            Input('dual-cursor-toggle', 'value'),
            State('cursor-select', 'value'),
            State('cursor-store', 'data'),
            State('main-graph', 'figure'),
            prevent_initial_call=True,
        )
        def update_cursors(click_data, dual_mode_value, cursor_select,
                           store_data, fig_data):
            triggered_id = callback_context.triggered_id
            dual_mode = bool(dual_mode_value and 'on' in dual_mode_value)

            if triggered_id == 'dual-cursor-toggle':
                if not dual_mode:
                    fig_data['layout']['shapes'] = []
                    store_data = {'1': {}, '2': {}}
                    return fig_data, '', store_data
                return fig_data, '', store_data

            if not dual_mode or click_data is None:
                return fig_data, '', store_data

            point = click_data['points'][0]
            x_clicked = point['x']
            x_axis = point.get('xaxis', 'x')

            cursor_key = cursor_select
            store_data[cursor_key][x_axis] = {
                'x': x_clicked,
                'y': point.get('y'),
            }

            shapes = list(fig_data['layout'].get('shapes', []))
            shapes = [
                s for s in shapes
                if not (
                    s.get('xref') == x_axis
                    and s.get('meta', {}).get('cursor') == cursor_key
                )
            ]

            y_axis = x_axis.replace('x', 'y')
            color = 'red' if cursor_key == '1' else 'blue'
            shapes.append({
                'type': 'line',
                'x0': x_clicked,
                'x1': x_clicked,
                'y0': 0,
                'y1': 1,
                'xref': x_axis,
                'yref': f'{y_axis} domain',
                'line': {'color': color, 'width': 1.5, 'dash': 'dash'},
                'meta': {'cursor': cursor_key},
            })

            fig_data['layout']['shapes'] = shapes

            info_lines = []
            for ckey, color_name in [('1', 'Cursor1'), ('2', 'Cursor2')]:
                for ax_name, pos in store_data[ckey].items():
                    y_val = pos.get('y')
                    y_str = f", y={y_val:.4f}" if y_val is not None else ""
                    info_lines.append(
                        f"{color_name} ({ax_name}): x={pos['x']:.4f}{y_str}")

            for ax_name in store_data['1']:
                if ax_name in store_data['2']:
                    dx = abs(store_data['2'][ax_name]['x']
                             - store_data['1'][ax_name]['x'])
                    info_lines.append(f"Δx ({ax_name}) = {dx:.4f}")

            return fig_data, '\n'.join(info_lines), store_data

        print(f"Dash app running at http://127.0.0.1:{port}/")
        app.run(port=port, debug=debug)

    def plot(self, suptitle="", dump_file_path=None, port=8050, debug=False):
        """
        Plots the simulation data using Plotly and Dash.

        Args:
            suptitle (str, optional): The title for the entire figure.
             Defaults to an empty string.
            dump_file_path (str, optional): Path to a dump file or directory
             containing dump files. If None, plots the current instance.
             Defaults to None.
            port (int, optional): Port number for the Dash server.
             Defaults to 8050.
            debug (bool, optional): Run Dash in debug mode.
             Defaults to False.
        """
        if dump_file_path is None:
            if self.activate_dump:
                self._dump_simulation_plotter()

            fig, shape = self._build_figure(suptitle)
            if fig is None:
                return

            self._run_dash_app(fig, shape, port=port, debug=debug)
            return

        path = dump_file_path
        if os.path.isdir(path):
            npz_files = [os.path.join(path, f) for f in os.listdir(path)
                         if f.lower().endswith('.npz')]
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
                loaded.plot(suptitle, port=port, debug=debug)
            except Exception as e:
                print(f"Failed to call plot() on loaded object: {e}")
            return

        if isinstance(loaded, dict):
            sp = SimulationPlotterDash(activate_dump=False)
            for k, v in loaded.items():
                try:
                    setattr(sp, k, v)
                except Exception:
                    pass
            try:
                sp.plot(suptitle, port=port, debug=debug)
            except Exception as e:
                print(
                    f"Failed to plot reconstructed SimulationPlotterDash: {e}")
            return

        print("Loaded dump does not contain a usable "
              "SimulationPlotterDash object.")
