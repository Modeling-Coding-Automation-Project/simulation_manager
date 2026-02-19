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

HOST_NAME = "0.0.0.0"
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
        self._pre_plot_figures = []

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

        signal_copy = np.copy(signal_object)
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_copy)
        else:
            self.name_to_object_dictionary[object_name] = [signal_copy]

    def append_name(self, signal_object, object_name):
        """
        Appends a signal object to the list associated with the given
        object name in the name_to_object_dictionary.

        Args:
            signal_object: The signal object to be associated with the object name.
            object_name (str): The key representing the name to which
             the signal object should be appended.
        """
        signal_copy = np.copy(signal_object)
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_copy)
        else:
            self.name_to_object_dictionary[object_name] = [signal_copy]

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
            signal_copy = np.copy(signal_sequence_object[i])
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_copy.reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_copy.reshape(-1, 1)]

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

    def pre_plot(self, suptitle=""):
        """
        Prepares a figure from the current subplot configuration
        and stores it internally for later display.

        This is the Dash equivalent of SimulationPlotter.pre_plot().
        Each call builds a Plotly figure from the currently assigned
        signals, stores it as a new tab page, and then resets the
        subplot assignment list so that subsequent assign / assign_all
        calls populate a new page.

        Call plot() after one or more pre_plot() calls to launch the
        Dash server and display all pages as browser tabs.

        Args:
            suptitle (str, optional): The title for this figure page.
                Defaults to an empty string.
        """
        if self.activate_dump:
            self._dump_simulation_plotter()

        fig, shape = self._build_figure(suptitle)
        if fig is None:
            return

        tab_label = suptitle if suptitle else f"Page {len(self._pre_plot_figures) + 1}"
        self._pre_plot_figures.append({
            'label': tab_label,
            'figure': fig,
            'shape': shape,
        })

        # Reset subplot assignments for the next page
        self.configuration.subplots_signals_list = []
        self.configuration.subplots_shape = np.zeros((2, 1), dtype=int)

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
                    arr = signal_object_list[i]
                    if arr.ndim == 0:
                        signal[i] = arr.item()
                    elif arr.ndim == 1:
                        signal[i] = arr[signal_info.column]
                    else:
                        signal[i] = arr[signal_info.column, signal_info.row]
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
                showspikes=True,
                spikemode='across',
                spikecolor='gray',
                spikethickness=1,
                spikedash='dot',
                row=plot_row,
                col=plot_col,
            )
            fig.update_yaxes(
                showgrid=True,
                showspikes=True,
                spikemode='across',
                spikecolor='gray',
                spikethickness=1,
                spikedash='dot',
                row=plot_row,
                col=plot_col,
            )

        fig.update_layout(
            title_text=suptitle,
            height=max(400, 300 * n_rows),
            showlegend=True,
        )

        return fig, shape

    def _run_dash_app(self, fig, shape, port=8050, debug=False,
                      tab_figures=None):
        """
        Launch the Dash application with the given Plotly figure(s).

        When *tab_figures* is provided (a list of dicts with keys
        'label', 'figure', 'shape'), the app renders each figure
        inside a separate Dash tab.  When *tab_figures* is ``None``
        the app shows a single figure (backward-compatible behaviour).

        Args:
            fig: The Plotly Figure object to display (single-tab mode).
            shape: A (2,1) numpy array with [n_rows, n_cols].
            port (int): Port number for the Dash server. Defaults to 8050.
            debug (bool): Run Dash in debug mode. Defaults to False.
            tab_figures (list[dict] | None): Optional list of figure
                pages to display in tabs.
        """
        app = Dash(__name__)

        # ---- build pages list ------------------------------------------
        if tab_figures and len(tab_figures) > 1:
            pages = tab_figures
        else:
            # single figure – wrap for uniform handling
            label = (tab_figures[0]['label']
                     if tab_figures else "Plot")
            pages = [{'label': label, 'figure': fig, 'shape': shape}]

        use_tabs = len(pages) > 1

        # ---- helper: build the content block for one page ---------------
        def _page_content(page_info, idx):
            graph_height = max(400, 300 * int(page_info['shape'][0, 0]))
            suffix = f"-{idx}" if use_tabs else ""
            return html.Div([
                html.Div(
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '20px',
                        'padding': '3px 12px',
                        'backgroundColor': '#f5f5f5',
                        'borderBottom': '1px solid #ddd',
                    },
                    children=[
                        dcc.Checklist(
                            id=f'dual-cursor-toggle{suffix}',
                            options=[{'label': ' Dual cursor mode',
                                      'value': 'on'}],
                            value=[],
                            style={'fontSize': '14px', 'lineHeight': '1'},
                            inputStyle={'margin': '0',
                                        'verticalAlign': 'middle'},
                            labelStyle={
                                'display': 'inline-flex',
                                'alignItems': 'center',
                                'gap': '4px',
                                'margin': '0',
                            },
                        ),
                        html.Div(
                            id=f'cursor-controls{suffix}',
                            children=[
                                html.Span('Select cursor: ',
                                          style={'fontSize': '13px'}),
                                dcc.RadioItems(
                                    id=f'cursor-select{suffix}',
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
                dcc.Graph(
                    id=f'main-graph{suffix}',
                    figure=page_info['figure'],
                    style={'height': f'{graph_height}px'},
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                    },
                ),
                dcc.Store(
                    id=f'cursor-store{suffix}',
                    data={'1': {}, '2': {}},
                ),
            ])

        # ---- layout ----------------------------------------------------
        if use_tabs:
            tabs_children = []
            for idx, page_info in enumerate(pages):
                tabs_children.append(
                    dcc.Tab(
                        label=page_info['label'],
                        children=[_page_content(page_info, idx)],
                        style={'padding': '6px 16px'},
                        selected_style={
                            'padding': '6px 16px',
                            'fontWeight': 'bold',
                            'borderTop': '3px solid #1f77b4',
                        },
                    )
                )
            app.layout = html.Div([
                dcc.Tabs(
                    id='page-tabs',
                    children=tabs_children,
                    value=None,
                ),
            ])
        else:
            app.layout = html.Div([
                _page_content(pages[0], 0),
            ])

        # ---- register callbacks for each page --------------------------
        for idx in range(len(pages)):
            suffix = f"-{idx}" if use_tabs else ""
            self._register_page_callbacks(
                app, suffix=suffix)

        print(f"Dash app running at http://127.0.0.1:{port}/")
        app.run(
            host=HOST_NAME,
            port=port,
            debug=debug)

    @staticmethod
    def _register_page_callbacks(app, suffix=""):
        """
        Register dual-cursor callbacks for a single page.

        Args:
            app: The Dash application instance.
            suffix (str): Suffix appended to component IDs to make
                them unique across tabs (e.g., "-0", "-1").
        """

        @app.callback(
            Output(f'cursor-controls{suffix}', 'style'),
            Input(f'dual-cursor-toggle{suffix}', 'value'),
        )
        def toggle_cursor_controls(dual_mode_value):
            if dual_mode_value and 'on' in dual_mode_value:
                return {'display': 'flex', 'alignItems': 'center', 'gap': '8px'}
            return {'display': 'none'}

        @app.callback(
            Output(f'main-graph{suffix}', 'figure'),
            Output(f'cursor-store{suffix}', 'data'),
            Input(f'main-graph{suffix}', 'clickData'),
            Input(f'dual-cursor-toggle{suffix}', 'value'),
            State(f'cursor-select{suffix}', 'value'),
            State(f'cursor-store{suffix}', 'data'),
            State(f'main-graph{suffix}', 'figure'),
            prevent_initial_call=True,
        )
        def update_cursors(click_data, dual_mode_value, cursor_select,
                           store_data, fig_data):
            triggered_id = callback_context.triggered_id
            toggle_id = f'dual-cursor-toggle{suffix}'
            dual_mode = bool(dual_mode_value and 'on' in dual_mode_value)

            if triggered_id == toggle_id:
                if not dual_mode:
                    fig_data['layout']['shapes'] = []
                    existing_ann = list(
                        fig_data['layout'].get('annotations', []))
                    fig_data['layout']['annotations'] = [
                        a for a in existing_ann
                        if not (a.get('name') or '').startswith(
                            'cursor_info_')]
                    store_data = {'1': {}, '2': {}}
                    return fig_data, store_data
                return fig_data, store_data

            if not dual_mode or click_data is None:
                return fig_data, store_data

            point = click_data['points'][0]
            x_clicked = point['x']

            # Determine the axes from the clicked trace's curveNumber,
            # because clickData does not always include xaxis/yaxis.
            curve_num = point.get('curveNumber', 0)
            trace_data = fig_data['data'][curve_num]
            x_axis = trace_data.get('xaxis', 'x')
            y_axis = trace_data.get('yaxis', 'y')

            cursor_key = cursor_select
            store_data[cursor_key][x_axis] = {
                'x': x_clicked,
                'y': point.get('y'),
                'y_axis': y_axis,
            }

            shapes = list(fig_data['layout'].get('shapes', []))
            shape_name = f'cursor_{cursor_key}_{x_axis}'
            shapes = [
                s for s in shapes
                if s.get('name', '') != shape_name
            ]

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
                'name': shape_name,
            })

            fig_data['layout']['shapes'] = shapes

            # Build per-subplot cursor info annotations
            existing_ann = list(
                fig_data['layout'].get('annotations', []))
            preserved_ann = [
                a for a in existing_ann
                if not (a.get('name') or '').startswith('cursor_info_')]

            all_x_axes = set()
            for ckey in ['1', '2']:
                all_x_axes.update(store_data[ckey].keys())

            for ax_name in sorted(all_x_axes):
                y_ax = None
                for ckey in ['1', '2']:
                    if ax_name in store_data[ckey]:
                        y_ax = store_data[ckey][ax_name].get('y_axis')
                        if y_ax:
                            break
                if y_ax is None:
                    continue

                text_parts = []
                for ckey, label, clr in [('1', 'C1', 'red'),
                                         ('2', 'C2', 'blue')]:
                    if ax_name in store_data[ckey]:
                        pos = store_data[ckey][ax_name]
                        y_val = pos.get('y')
                        y_str = (f", y={y_val:.4f}"
                                 if y_val is not None else "")
                        text_parts.append(
                            f'<span style="color:{clr}">'
                            f'{label}: x={pos["x"]:.4f}{y_str}</span>')
                if (ax_name in store_data['1']
                        and ax_name in store_data['2']):
                    dx = abs(store_data['2'][ax_name]['x']
                             - store_data['1'][ax_name]['x'])
                    text_parts.append(f'\u0394x={dx:.4f}')

                if text_parts:
                    preserved_ann.append({
                        'name': f'cursor_info_{ax_name}',
                        'text': '<br>'.join(text_parts),
                        'xref': f'{ax_name} domain',
                        'yref': f'{y_ax} domain',
                        'x': 0.01,
                        'y': 0.99,
                        'xanchor': 'left',
                        'yanchor': 'top',
                        'showarrow': False,
                        'font': {'size': 11, 'family': 'monospace',
                                 'color': '#333'},
                        'bgcolor': 'rgba(255,251,230,0.9)',
                        'bordercolor': '#ccc',
                        'borderwidth': 1,
                        'borderpad': 4,
                    })

            fig_data['layout']['annotations'] = preserved_ann

            return fig_data, store_data

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
            # Build a figure from any remaining assigned signals
            remaining_fig, remaining_shape = None, None
            if self.configuration.subplots_signals_list:
                if self.activate_dump:
                    self._dump_simulation_plotter()
                remaining_fig, remaining_shape = self._build_figure(suptitle)

            # Collect all pre_plot pages + the remaining figure
            all_pages = list(self._pre_plot_figures)
            if remaining_fig is not None:
                tab_label = (suptitle if suptitle
                             else f"Page {len(all_pages) + 1}")
                all_pages.append({
                    'label': tab_label,
                    'figure': remaining_fig,
                    'shape': remaining_shape,
                })

            if not all_pages:
                print("No subplots to show.")
                return

            # Use first page as the base fig/shape for backward compat
            fig = all_pages[0]['figure']
            shape = all_pages[0]['shape']

            self._run_dash_app(fig, shape, port=port, debug=debug,
                               tab_figures=all_pages)
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
