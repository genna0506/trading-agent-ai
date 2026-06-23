"""Grafici Plotly per i risultati del backtest (Fase 2).

Coerenti con lo stile di :mod:`src.analysis.charts`: ogni funzione restituisce
una ``Figure`` (senza ``.show()``), riutilizzabile in notebook o dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.analysis.metrics import max_drawdown
from src.backtest.engine import BacktestResult

_TEMPLATE = "plotly_white"
_COLOR_EQUITY = "#2E86DE"
_COLOR_DD = "rgba(231, 76, 60, 0.18)"


def plot_equity_curve(result: BacktestResult) -> go.Figure:
    """Equity curve della strategia con shading del drawdown.

    Parameters
    ----------
    result:
        Output di :func:`src.backtest.engine.run_backtest`.

    Returns
    -------
    go.Figure
    """
    equity = result.equity.dropna()
    running_max = equity.cummax()
    mdd = max_drawdown(equity)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=running_max.index, y=running_max.values,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            fill="tonexty", fillcolor=_COLOR_DD,
            line=dict(color=_COLOR_EQUITY, width=2),
            name=result.name,
            hovertemplate="%{x|%d %b %Y}<br>€%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Equity curve — {result.name} (Max Drawdown: {mdd:.1%})",
        xaxis_title="Data", yaxis_title="Valore (EUR)",
        template=_TEMPLATE, hovermode="x unified",
    )
    return fig


def plot_strategy_comparison(results: dict[str, BacktestResult]) -> go.Figure:
    """Confronta più equity curve normalizzate a base 100.

    Parameters
    ----------
    results:
        Mappa ``{nome: BacktestResult}`` (es. output di
        :func:`src.backtest.engine.compare_strategies`).

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for name, res in results.items():
        equity = res.equity.dropna()
        if equity.empty:
            continue
        normalized = 100.0 * equity / equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=normalized.index, y=normalized.values, name=name,
                hovertemplate=f"{name}<br>%{{x|%d %b %Y}}<br>%{{y:.1f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Confronto strategie (base 100)",
        xaxis_title="Data", yaxis_title="Indice (base 100)",
        template=_TEMPLATE, hovermode="x unified",
    )
    return fig
