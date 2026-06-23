"""Motore di backtest — STUB Fase 2.

Punto di estensione per testare strategie sulle serie storiche prodotte da
:func:`src.analysis.metrics.portfolio_value_over_time`. L'interfaccia è
volutamente minimale: l'implementazione arriverà nella Fase 2.
"""

from __future__ import annotations

import pandas as pd


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Esegue un backtest di una strategia (NON ancora implementato).

    Parameters
    ----------
    signals:
        Segnali di posizione per ticker e data.
    prices:
        Prezzi storici allineati ai segnali.
    initial_capital:
        Capitale iniziale in EUR.

    Returns
    -------
    pd.DataFrame
        Equity curve e statistiche della strategia.
    """
    raise NotImplementedError("Il motore di backtest sarà implementato nella Fase 2.")
