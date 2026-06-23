"""Visualizzazioni interattive del portfolio con Plotly.

Ogni funzione restituisce un ``plotly.graph_objects.Figure`` (non chiama
``.show()``) così le figure possono essere mostrate inline nei notebook,
incorporate nella dashboard Streamlit (Fase 4) o salvate su file.

Le funzioni accettano gli output dei moduli :mod:`src.analysis.metrics` per
mantenere una netta separazione tra calcolo e presentazione.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis.metrics import max_drawdown

# Palette coerente fra i grafici.
_COLOR_LINE = "#2E86DE"
_COLOR_DRAWDOWN = "rgba(231, 76, 60, 0.18)"
_COLOR_POS = "#27AE60"
_COLOR_NEG = "#E74C3C"
_TEMPLATE = "plotly_white"


def plot_portfolio_value(
    df: pd.DataFrame, *, value_col: str = "holdings_value"
) -> go.Figure:
    """Grafico a linea del valore del portfolio con shading dei drawdown.

    Le aree di drawdown (sotto il massimo storico) sono evidenziate riempiendo
    lo spazio tra la curva del valore e la curva dei massimi correnti.

    Parameters
    ----------
    df:
        Output di :func:`metrics.portfolio_value_over_time` (indice = date).
    value_col:
        Colonna da plottare (default ``total_value``).

    Returns
    -------
    go.Figure
    """
    series = df[value_col].dropna()
    running_max = series.cummax()
    mdd = max_drawdown(series)

    fig = go.Figure()
    # Curva dei massimi (invisibile, fa da bordo superiore al fill).
    fig.add_trace(
        go.Scatter(
            x=running_max.index,
            y=running_max.values,
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Area di drawdown: dal massimo corrente giù fino al valore.
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            fill="tonexty",
            fillcolor=_COLOR_DRAWDOWN,
            line=dict(color=_COLOR_LINE, width=2),
            name="Valore portfolio",
            hovertemplate="%{x|%d %b %Y}<br>€%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Valore del portfolio nel tempo (Max Drawdown: {mdd:.1%})",
        xaxis_title="Data",
        yaxis_title="Valore (EUR)",
        template=_TEMPLATE,
        hovermode="x unified",
    )
    return fig


def plot_allocation_pie(df: pd.DataFrame) -> go.Figure:
    """Grafico a torta (donut) interattivo dell'allocazione attuale.

    Parameters
    ----------
    df:
        Output di :func:`metrics.allocation_breakdown` (indice = ticker, con
        colonne ``market_value_eur`` e ``weight_pct``).

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(
        go.Pie(
            labels=df.index.tolist(),
            values=df["market_value_eur"].clip(lower=0).tolist(),
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}<br>€%{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Allocazione attuale del portfolio",
        template=_TEMPLATE,
    )
    return fig


def plot_pnl_waterfall(df: pd.DataFrame) -> go.Figure:
    """Waterfall chart del P&L realizzato per ticker.

    Parameters
    ----------
    df:
        Output di :func:`metrics.realized_pnl` (indice = ticker, colonna
        ``realized_pnl_eur``).

    Returns
    -------
    go.Figure
    """
    ordered = df.sort_values("realized_pnl_eur", ascending=False)
    values = ordered["realized_pnl_eur"].tolist()
    labels = ordered.index.tolist()

    fig = go.Figure(
        go.Waterfall(
            x=labels + ["Totale"],
            y=values + [sum(values)],
            measure=["relative"] * len(values) + ["total"],
            connector={"line": {"color": "rgba(120,120,120,0.4)"}},
            increasing={"marker": {"color": _COLOR_POS}},
            decreasing={"marker": {"color": _COLOR_NEG}},
            totals={"marker": {"color": _COLOR_LINE}},
            hovertemplate="%{x}<br>€%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="P&L realizzato per ticker (FIFO)",
        xaxis_title="Ticker",
        yaxis_title="P&L realizzato (EUR)",
        template=_TEMPLATE,
    )
    return fig


def plot_monthly_returns_heatmap(
    df: pd.DataFrame, *, value_col: str = "holdings_value"
) -> go.Figure:
    """Heatmap calendario dei rendimenti mensili (anni x mesi).

    I rendimenti mensili sono calcolati dall'ultimo valore di portfolio di ogni
    mese.

    Parameters
    ----------
    df:
        Output di :func:`metrics.portfolio_value_over_time` (indice = date).
    value_col:
        Colonna del valore da usare.

    Returns
    -------
    go.Figure
    """
    series = df[value_col].dropna()
    monthly = series.resample("ME").last()
    returns = monthly.pct_change().dropna()

    if returns.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Rendimenti mensili (dati insufficienti)", template=_TEMPLATE
        )
        return fig

    table = pd.DataFrame(
        {
            "year": returns.index.year,
            "month": returns.index.month,
            "ret": returns.values * 100.0,
        }
    )
    pivot = table.pivot(index="year", columns="month", values="ret").reindex(
        columns=range(1, 13)
    )
    month_labels = [
        "Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
        "Lug", "Ago", "Set", "Ott", "Nov", "Dic",
    ]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index.astype(str).tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            hovertemplate="%{y} %{x}<br>%{z:.2f}%<extra></extra>",
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        title="Heatmap rendimenti mensili",
        xaxis_title="Mese",
        yaxis_title="Anno",
        template=_TEMPLATE,
    )
    return fig
