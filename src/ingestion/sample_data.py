"""Generatore di transazioni di esempio nello schema canonico.

Utile finché i CSV reali dei broker non sono disponibili: alimenta il notebook
di esplorazione e i test con dati deterministici che imitano il formato reale
(acquisti, vendite parziali, dividendi, commissioni, valuta estera).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRANSACTION_COLUMNS


def make_sample_transactions(seed: int = 42) -> pd.DataFrame:
    """Crea un DataFrame canonico di transazioni di esempio (deterministico).

    Include:
    * più acquisti dello stesso ticker su lotti diversi e una vendita parziale
      (per esercitare il matching FIFO multi-lotto);
    * una posizione in USD (per la conversione FX);
    * dividendi, commissioni e imposte.

    Parameters
    ----------
    seed:
        Seme per eventuali componenti casuali (qui i dati sono per lo più fissi).

    Returns
    -------
    pd.DataFrame
        Transazioni con le colonne di :data:`TRANSACTION_COLUMNS`.
    """
    _ = np.random.default_rng(seed)  # placeholder per estensioni future

    rows = [
        # ticker, type, date, qty, price, currency, fx_rate
        ("VWCE.DE", "buy", "2023-01-10", 50, 100.0, "EUR", 1.0),
        ("VWCE.DE", "buy", "2023-04-15", 30, 110.0, "EUR", 1.0),
        ("VWCE.DE", "sell", "2023-09-20", 40, 120.0, "EUR", 1.0),  # vendita parziale
        ("AAPL", "buy", "2023-02-01", 20, 150.0, "USD", 0.92),
        ("AAPL", "buy", "2023-06-01", 10, 180.0, "USD", 0.93),
        ("AAPL", "dividend", "2023-08-15", 0, 0.0, "USD", 0.91),
        ("ENI.MI", "buy", "2023-03-05", 100, 13.5, "EUR", 1.0),
        ("ENI.MI", "dividend", "2023-05-20", 0, 0.0, "EUR", 1.0),
    ]

    records = []
    for ticker, ttype, dt, qty, price, currency, fx in rows:
        if ttype == "buy":
            amount = -qty * price * fx          # uscita di cassa
        elif ttype == "sell":
            amount = qty * price * fx           # entrata di cassa
        elif ttype == "dividend":
            amount = {"AAPL": 25.0, "ENI.MI": 45.0}.get(ticker, 10.0) * fx
        else:
            amount = -5.0
        records.append(
            {
                "date": pd.Timestamp(dt),
                "ticker": ticker,
                "type": ttype,
                "quantity": float(qty),
                "price": float(price),
                "amount_eur": float(amount),
                "currency": currency,
                "fx_rate": float(fx),
                "broker": "sample",
            }
        )

    # Qualche commissione/imposta sparsa.
    records += [
        {
            "date": pd.Timestamp("2023-01-10"),
            "ticker": "CASH",
            "type": "fee",
            "quantity": np.nan,
            "price": np.nan,
            "amount_eur": -7.5,
            "currency": "EUR",
            "fx_rate": 1.0,
            "broker": "sample",
        },
        {
            "date": pd.Timestamp("2023-09-20"),
            "ticker": "CASH",
            "type": "tax",
            "quantity": np.nan,
            "price": np.nan,
            "amount_eur": -32.0,
            "currency": "EUR",
            "fx_rate": 1.0,
            "broker": "sample",
        },
    ]

    df = pd.DataFrame.from_records(records)
    return df.sort_values("date").reset_index(drop=True)[list(TRANSACTION_COLUMNS)]


def make_sample_price_history(
    transactions_df: pd.DataFrame, *, seed: int = 7
) -> pd.DataFrame:
    """Genera uno storico prezzi sintetico per i ticker delle transazioni.

    Random walk geometrico deterministico, indicizzato giornalmente dalla prima
    transazione a oggi. Pensato per alimentare
    :func:`metrics.portfolio_value_over_time` offline.
    """
    rng = np.random.default_rng(seed)
    tickers = sorted(
        transactions_df.loc[
            transactions_df["type"].isin(["buy", "sell"]), "ticker"
        ].unique()
    )
    start = transactions_df["date"].min().normalize()
    end = pd.Timestamp.today().normalize()
    idx = pd.date_range(start, end, freq="D")

    data = {}
    for tk in tickers:
        start_price = float(
            transactions_df.loc[transactions_df["ticker"].eq(tk), "price"].iloc[0]
        ) or 100.0
        steps = rng.normal(0.0003, 0.012, size=len(idx))
        data[tk] = start_price * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=idx)
