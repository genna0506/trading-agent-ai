"""Motore di backtest vettorizzato (Fase 2).

Simula l'andamento di un portafoglio dati i prezzi storici e una serie di
**pesi target** (prodotti da una :class:`~src.backtest.strategies.Strategy`).
La simulazione è interamente vettorizzata (nessun loop riga-per-riga) e applica
i costi di transazione sul *turnover* (variazione dei pesi).

Per non introdurre *look-ahead bias*, il rendimento di un giorno è guadagnato
con i pesi del giorno **precedente** (si decide oggi, si incassa domani).

Le metriche di performance riusano direttamente la Fase 1
(:func:`src.analysis.metrics.sharpe_ratio`, :func:`~.max_drawdown`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.metrics import TRADING_DAYS_PER_YEAR, max_drawdown, sharpe_ratio
from src.backtest.strategies import Strategy


@dataclass
class BacktestResult:
    """Risultato di un backtest.

    Attributes
    ----------
    equity:
        Valore del portafoglio nel tempo (serie indicizzata per data).
    returns:
        Rendimenti giornalieri netti (al netto dei costi).
    weights:
        Pesi target applicati (date x ticker).
    initial_capital:
        Capitale iniziale in EUR.
    fee_bps:
        Costo di transazione in basis point sul turnover.
    name:
        Nome della strategia.
    """

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    initial_capital: float
    fee_bps: float
    name: str = "strategy"

    def summary(self, risk_free_rate: float = 0.03) -> dict[str, float]:
        """Statistiche di sintesi della performance.

        Returns
        -------
        dict
            ``total_return``, ``cagr``, ``sharpe``, ``max_drawdown``,
            ``volatility`` (annua), ``n_rebalances``.
        """
        equity = self.equity.dropna()
        if len(equity) < 2:
            return {
                "total_return": float("nan"),
                "cagr": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "volatility": float("nan"),
                "n_rebalances": 0.0,
            }
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else float("nan")
        volatility = float(self.returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        turnover = self.weights.diff().abs().sum(axis=1)
        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe_ratio(equity, risk_free_rate=risk_free_rate),
            "max_drawdown": max_drawdown(equity),
            "volatility": volatility,
            "n_rebalances": float((turnover > 1e-9).sum()),
        }


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
    fee_bps: float = 0.0,
    name: str = "strategy",
) -> BacktestResult:
    """Esegue un backtest vettorizzato dati prezzi e pesi target.

    Parameters
    ----------
    prices:
        Prezzi di chiusura storici (indice = date, colonne = ticker).
    weights:
        Pesi target in ``[0, 1]`` (date x ticker). Vengono allineati a
        ``prices`` (reindex + fill a 0).
    initial_capital:
        Capitale iniziale in EUR.
    fee_bps:
        Costo di transazione in basis point applicato al turnover giornaliero
        (es. ``fee_bps=10`` = 0,10% sul controvalore ribilanciato).
    name:
        Nome della strategia (per report/grafici).

    Returns
    -------
    BacktestResult
    """
    prices = prices.sort_index()
    asset_returns = prices.pct_change().fillna(0.0)

    weights = (
        weights.reindex(index=prices.index, columns=prices.columns)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # Rendimento di oggi guadagnato con i pesi di ieri (no look-ahead).
    lagged = weights.shift(1).fillna(0.0)
    gross_returns = (lagged * asset_returns).sum(axis=1)

    # Costi: turnover (variazione assoluta dei pesi) * fee.
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    costs = turnover * (fee_bps / 10_000.0)

    net_returns = (gross_returns - costs).rename("returns")
    equity = (initial_capital * (1.0 + net_returns).cumprod()).rename("equity")

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        weights=weights,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        name=name,
    )


def backtest_strategy(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    initial_capital: float = 10_000.0,
    fee_bps: float = 0.0,
) -> BacktestResult:
    """Comodità: genera i pesi dalla strategia ed esegue il backtest.

    Parameters
    ----------
    prices:
        Prezzi storici.
    strategy:
        Istanza di :class:`~src.backtest.strategies.Strategy`.
    initial_capital, fee_bps:
        Parametri passati a :func:`run_backtest`.
    """
    weights = strategy.generate_weights(prices)
    return run_backtest(
        prices,
        weights,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        name=strategy.name,
    )


def compare_strategies(
    prices: pd.DataFrame,
    strategies: list[Strategy],
    *,
    initial_capital: float = 10_000.0,
    fee_bps: float = 0.0,
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Esegue più strategie sugli stessi prezzi e ne confronta le statistiche.

    Returns
    -------
    results : dict[str, BacktestResult]
        Risultati per nome strategia.
    summary_table : pd.DataFrame
        Tabella con una riga per strategia e le statistiche di :meth:`summary`.
    """
    results = {
        strat.name: backtest_strategy(
            prices, strat, initial_capital=initial_capital, fee_bps=fee_bps
        )
        for strat in strategies
    }
    summary_table = pd.DataFrame(
        {name: res.summary() for name, res in results.items()}
    ).T
    return results, summary_table
