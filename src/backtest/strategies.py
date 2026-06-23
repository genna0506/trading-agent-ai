"""Strategie di investimento per il motore di backtest (Fase 2).

Una *strategia* è un oggetto con un metodo
``generate_weights(prices) -> pd.DataFrame`` che, dati i prezzi storici
(indice = date, colonne = ticker), restituisce i **pesi target** del
portafoglio per ogni giorno: un valore in ``[0, 1]`` per ticker che indica la
frazione di capitale da allocare. La somma dei pesi per riga è ``<= 1`` (la
parte restante è liquidità).

Il motore (:mod:`src.backtest.engine`) è agnostico rispetto alla strategia:
gli basta ricevere un DataFrame di pesi. Così aggiungere nuove strategie non
richiede modifiche al motore.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Interfaccia base di una strategia di allocazione."""

    #: Nome leggibile, usato nei grafici e nei report.
    name: str = "strategy"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calcola i pesi target del portafoglio per ogni data.

        Parameters
        ----------
        prices:
            Prezzi di chiusura storici (indice = date, colonne = ticker).

        Returns
        -------
        pd.DataFrame
            Pesi in ``[0, 1]`` (date x ticker), allineati a ``prices``.
        """
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Compra all'inizio in equal-weight e mantiene (benchmark di riferimento).

    Alloca ``1/N`` del capitale su ciascuno degli ``N`` ticker e non ribilancia
    mai i pesi *target* (i pesi effettivi driftano coi prezzi, ma per un
    benchmark equal-weight buy&hold questa approssimazione è standard).
    """

    name = "Buy & Hold"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        n_assets = prices.shape[1]
        weight = 1.0 / n_assets if n_assets else 0.0
        return pd.DataFrame(weight, index=prices.index, columns=prices.columns)


class SMACrossover(Strategy):
    """Incrocio di medie mobili (trend-following) per singolo asset.

    Per ogni ticker è "investito" (peso ``1/N``) nei giorni in cui la media
    mobile veloce è sopra quella lenta, altrimenti è in liquidità (peso 0).

    Parameters
    ----------
    fast:
        Finestra della media mobile veloce (in giorni).
    slow:
        Finestra della media mobile lenta (in giorni).
    """

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("`fast` deve essere minore di `slow`")
        self.fast = fast
        self.slow = slow
        self.name = f"SMA {fast}/{slow}"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast_ma = prices.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = prices.rolling(self.slow, min_periods=self.slow).mean()
        # Segnale binario 1/0; NaN (warmup) -> 0 (fuori dal mercato).
        signal = (fast_ma > slow_ma).astype(float)
        signal = signal.where(slow_ma.notna(), 0.0)
        n_assets = prices.shape[1]
        return signal / n_assets if n_assets else signal


class Momentum(Strategy):
    """Momentum cross-sectional: investe negli asset col trend più forte.

    Ogni giorno calcola il rendimento degli ultimi ``lookback`` giorni e alloca
    equal-weight sugli asset con momentum positivo (peso 0 sugli altri).

    Parameters
    ----------
    lookback:
        Finestra del rendimento passato (in giorni).
    """

    def __init__(self, lookback: int = 60) -> None:
        if lookback < 1:
            raise ValueError("`lookback` deve essere >= 1")
        self.lookback = lookback
        self.name = f"Momentum {lookback}g"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        past_return = prices.pct_change(self.lookback)
        positive = (past_return > 0).astype(float)
        positive = positive.where(past_return.notna(), 0.0)
        # Equal-weight tra gli asset "on" di ciascun giorno; 0 se nessuno.
        n_on = positive.sum(axis=1).replace(0, np.nan)
        weights = positive.div(n_on, axis=0).fillna(0.0)
        return weights
