import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_ppg(signals, markers=None, x_interval=None, retfig=False):
    """
    Parameters
    ----------
        signals : DataFrame
            the signals to plot. Each column will be plotted as an individual signal.
            The index should be a range index (sample id)

        markers : DataFrame of dtype bool, optional
            Needs to have the same index and columns as signals.
            Where a value is True, a marker wil appear in the plot.
            This parameter can be used to mark spectral peaks.
        
        x_interval : list-like, optional
            Two integers used to preset an x axis zoom in the figure

        retfig : bool, optional
            if the resulting plotting figure should be returned

    Returns
    ----------
        figure : plotly.graph_objects.Figure
            Returned only if retfig is True
    """

    if isinstance(signals, np.ndarray):
        signals = pd.DataFrame(signals.T)
    
    color_map = dict(zip(signals.columns, px.colors.qualitative.Alphabet))
    fig = go.Figure()

    for name in signals.columns:
        fig.add_trace(go.Scatter(x=signals.index, y=signals[name], name=name, mode='lines',
                                 line = dict(color=color_map[name])))

    if markers:
        for name, idx_arr in markers.items():
            fig.add_trace(go.Scatter(x=idx_arr, y=signals.loc[idx_arr, name], mode='markers', 
                                     line = dict(color=color_map[name])))
            fig['data'][-1]['showlegend'] = False

    if x_interval is not None:
        fig.update_xaxes(range=x_interval)

    fig.update_layout(
        xaxis_title="sample id",
        yaxis_title="Amplitude"
    )

    if retfig:
        return fig
    
    fig.show()

def plot_spectrum(signals, markers=None, x_interval=None, retfig=False):
    """
    Parameters
    ----------
        signals : DataFrame or ndarray of shape (n_signals, n_freq_points)
            the spectra to plot. Each column will be plotted as an individual spectrum.
            The index values are interpreted as frequencies.

        markers : DataFrame of dtype bool, optional
            Needs to have the same index and columns as signals.
            Where a value is True, a marker wil appear in the plot.
            This parameter can be used to mark spectral peaks.
        
        x_interval : list-like, optional
            Two integers used to preset an x axis zoom in the figure

        retfig : bool, optional
            if the resulting plotting figure should be returned

    Returns
    ----------
        figure : plotly.graph_objects.Figure
            Returned only if retfig is True
    """
    if isinstance(signals, np.ndarray):
        signals = pd.DataFrame(signals.T)

    color_map = dict(zip(signals.columns, px.colors.qualitative.Alphabet))
    fig = go.Figure()

    for name in signals.columns:
        fig.add_trace(go.Scatter(x=signals.index, y=signals[name], name=name, mode='lines',
                                 line = dict(color=color_map[name])))

    if markers:
        for name, idx_arr in markers.items():
            fig.add_trace(go.Scatter(x=idx_arr, y=signals.loc[idx_arr, name], mode='markers', 
                                     line = dict(color=color_map[name])))
            fig['data'][-1]['showlegend'] = False

    if x_interval is not None:
        fig.update_xaxes(range=x_interval)

    fig.update_layout(
        xaxis_title="frequency",
        yaxis_title="amount"
    )

    if retfig:
        return fig
    
    fig.show()
