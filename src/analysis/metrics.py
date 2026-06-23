"""Metriche di portfolio: valore, P&L (FIFO), rendimenti, rischio, allocazione.

Tutte le funzioni accettano il DataFrame canonico prodotto da
:mod:`src.ingestion.loader` (schema :data:`src.config.TRANSACTION_COLUMNS`) e
sono progettate per essere chiamate direttamente — anche dagli strumenti
dell'agente AI della Fase 4.

Per testabilità offline, le funzioni che recuperano prezzi/quote accettano un
parametro opzionale (``price_data`` / ``current_prices``) per iniettare i dati
ed evitare chiamate di rete.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tipi che muovono quantità di titoli.
_POSITION_TYPES = ("buy", "sell")
TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# FIFO engine
# --------------------------------------------------------------------------- #
@dataclass
class _Lot:
    """Un lotto di acquisto residuo nella coda FIFO."""

    quantity: float
    cost_per_unit_eur: float


def _run_fifo(
    transactions_df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, list[_Lot]]]:
    """Esegue il matching FIFO di acquisti e vendite per ogni ticker.

    Per ogni vendita consuma i lotti di acquisto più vecchi, gestendo
    correttamente le vendite parziali distribuite su più lotti.

    Returns
    -------
    realized : pd.Series
        P&L realizzato in EUR per ticker.
    open_lots : dict[str, list[_Lot]]
        Lotti di acquisto residui (non ancora venduti) per ticker, usati dal
        calcolo dell'unrealized.
    """
    trades = (
        transactions_df[transactions_df["type"].isin(_POSITION_TYPES)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    lots: dict[str, deque[_Lot]] = defaultdict(deque)
    realized: dict[str, float] = defaultdict(float)

    # itertuples è ~10x più veloce di iterrows ed evita il boxing in Series.
    for tx in trades.itertuples(index=False):
        qty = abs(float(tx.quantity)) if not pd.isna(tx.quantity) else 0.0
        if qty == 0:
            continue
        # prezzo unitario in EUR: preferisci price*fx, altrimenti deriva da amount.
        if not pd.isna(tx.price) and tx.price:
            unit_eur = float(tx.price) * float(tx.fx_rate or 1.0)
        else:
            unit_eur = abs(float(tx.amount_eur)) / qty if qty else 0.0

        if tx.type == "buy":
            lots[tx.ticker].append(_Lot(quantity=qty, cost_per_unit_eur=unit_eur))
            continue

        # type == "sell": consuma i lotti più vecchi finché copri la quantità.
        remaining = qty
        queue = lots[tx.ticker]
        while remaining > 1e-12 and queue:
            lot = queue[0]
            matched = min(remaining, lot.quantity)
            realized[tx.ticker] += matched * (unit_eur - lot.cost_per_unit_eur)
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= 1e-12:
                queue.popleft()
        if remaining > 1e-9:
            # vendita allo scoperto / dati incompleti: registriamo comunque il
            # ricavo come realized con costo zero, segnalando l'anomalia.
            logger.warning(
                "Vendita di %s eccede i lotti disponibili di %.4f unità",
                tx.ticker,
                remaining,
            )
            realized[tx.ticker] += remaining * unit_eur

    realized_series = pd.Series(realized, dtype="float64").rename("realized_pnl_eur")
    realized_series.index.name = "ticker"
    open_lots = {t: list(q) for t, q in lots.items() if q}
    return realized_series, open_lots


def realized_pnl(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """P&L realizzato per ticker con metodo FIFO.

    Le vendite parziali vengono abbinate ai lotti di acquisto più vecchi; il
    costo di ciascuna unità venduta è quello del lotto da cui proviene.

    Parameters
    ----------
    transactions_df:
        DataFrame canonico delle transazioni.

    Returns
    -------
    pd.DataFrame
        Indicizzato per ``ticker`` con colonna ``realized_pnl_eur``, ordinato
        per P&L decrescente.
    """
    realized, _ = _run_fifo(transactions_df)
    if realized.empty:
        return pd.DataFrame(columns=["realized_pnl_eur"]).rename_axis("ticker")
    return realized.sort_values(ascending=False).to_frame()


def current_holdings(transactions_df: pd.DataFrame) -> pd.Series:
    """Quantità netta attualmente detenuta per ticker (buy - sell).

    Returns una Series indicizzata per ticker con la quantità > 0.
    """
    trades = transactions_df[transactions_df["type"].isin(_POSITION_TYPES)].copy()
    if trades.empty:
        return pd.Series(dtype="float64", name="quantity").rename_axis("ticker")
    signed = np.where(trades["type"].eq("buy"), 1.0, -1.0) * trades["quantity"].abs()
    holdings = (
        pd.Series(signed, index=trades.index)
        .groupby(trades["ticker"])
        .sum()
        .rename("quantity")
    )
    holdings.index.name = "ticker"
    return holdings[holdings.abs() > 1e-9]


# --------------------------------------------------------------------------- #
# Prezzi correnti / storici (yfinance, con iniezione per i test)
# --------------------------------------------------------------------------- #
def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Recupera l'ultimo prezzo di chiusura per ogni ticker via yfinance."""
    if not tickers:
        return {}
    import yfinance as yf  # import lazy

    prices: dict[str, float] = {}
    data = yf.download(
        tickers, period="5d", progress=False, auto_adjust=True, group_by="ticker"
    )
    for tk in tickers:
        try:
            series = data[tk]["Close"] if len(tickers) > 1 else data["Close"]
            last = series.dropna().iloc[-1]
            prices[tk] = float(last)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prezzo corrente non disponibile per %s (%s)", tk, exc)
    return prices


def unrealized_pnl(
    transactions_df: pd.DataFrame,
    *,
    current_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    """P&L non realizzato sulle posizioni aperte (mark-to-market).

    Per ogni ticker ancora in portafoglio calcola il costo dei lotti FIFO
    residui e lo confronta con il valore di mercato attuale.

    Parameters
    ----------
    transactions_df:
        DataFrame canonico.
    current_prices:
        Mappa ``{ticker: prezzo}`` per evitare la rete (utile nei test). Se
        ``None`` i prezzi vengono scaricati da yfinance.

    Returns
    -------
    pd.DataFrame
        Per ticker: ``quantity``, ``avg_cost_eur``, ``cost_basis_eur``,
        ``current_price``, ``market_value_eur``, ``unrealized_pnl_eur``.
    """
    _, open_lots = _run_fifo(transactions_df)
    if not open_lots:
        return pd.DataFrame(
            columns=[
                "quantity",
                "avg_cost_eur",
                "cost_basis_eur",
                "current_price",
                "market_value_eur",
                "unrealized_pnl_eur",
            ]
        ).rename_axis("ticker")

    rows = {}
    for ticker, lots in open_lots.items():
        qty = sum(lot.quantity for lot in lots)
        cost_basis = sum(lot.quantity * lot.cost_per_unit_eur for lot in lots)
        rows[ticker] = {
            "quantity": qty,
            "avg_cost_eur": cost_basis / qty if qty else np.nan,
            "cost_basis_eur": cost_basis,
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"

    if current_prices is None:
        current_prices = _fetch_current_prices(df.index.tolist())

    df["current_price"] = df.index.map(lambda t: current_prices.get(t, np.nan))
    df["market_value_eur"] = df["quantity"] * df["current_price"]
    df["unrealized_pnl_eur"] = df["market_value_eur"] - df["cost_basis_eur"]
    return df.sort_values("unrealized_pnl_eur", ascending=False)


# --------------------------------------------------------------------------- #
# Valore del portfolio nel tempo
# --------------------------------------------------------------------------- #
def _fetch_price_history(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Scarica i prezzi di chiusura storici (colonne = ticker)."""
    if not tickers:
        return pd.DataFrame()
    import yfinance as yf  # import lazy

    data = yf.download(
        tickers,
        start=start.normalize(),
        end=(end.normalize() + pd.Timedelta(days=1)),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    closes = {}
    for tk in tickers:
        try:
            closes[tk] = data[tk]["Close"] if len(tickers) > 1 else data["Close"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Storico prezzi non disponibile per %s (%s)", tk, exc)
    return pd.DataFrame(closes)


def portfolio_value_over_time(
    transactions_df: pd.DataFrame,
    *,
    price_data: pd.DataFrame | None = None,
    end: pd.Timestamp | date | None = None,
) -> pd.DataFrame:
    """Valore giornaliero del portfolio (mark-to-market) nel tempo.

    Costruisce le posizioni cumulate giornaliere per ticker e le valorizza con
    i prezzi storici. Aggiunge una componente di cassa data dal flusso netto
    cumulato delle transazioni (dividendi/vendite la aumentano, acquisti/costi
    la riducono).

    .. note::
       Poiché gli export di soli *trade* non contengono i versamenti di
       liquidità, ``cash`` (e quindi ``total_value``) può risultare negativa: è
       il saldo dei flussi *tracciati*, non un conto reale. Per le metriche di
       rischio (drawdown, Sharpe) usare ``holdings_value``, che rappresenta la
       curva onesta del valore di mercato delle posizioni.

    Parameters
    ----------
    transactions_df:
        DataFrame canonico.
    price_data:
        DataFrame di prezzi storici (indice = date, colonne = ticker) per
        evitare la rete nei test. Se ``None`` viene scaricato da yfinance.
    end:
        Data finale della serie (default: oggi).

    Returns
    -------
    pd.DataFrame
        Indicizzato per data con colonne ``holdings_value``, ``cash``,
        ``total_value``.
    """
    if transactions_df.empty:
        return pd.DataFrame(columns=["holdings_value", "cash", "total_value"])

    tx = transactions_df.sort_values("date").copy()
    start = pd.Timestamp(tx["date"].min()).normalize()
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    calendar = pd.date_range(start, end, freq="D")

    # Posizioni cumulate giornaliere per ticker (vettorizzato via pivot+cumsum).
    trades = tx[tx["type"].isin(_POSITION_TYPES)].copy()
    if trades.empty:
        holdings_value = pd.Series(0.0, index=calendar)
    else:
        trades["signed_qty"] = np.where(
            trades["type"].eq("buy"), 1.0, -1.0
        ) * trades["quantity"].abs()
        trades["day"] = trades["date"].dt.normalize()
        daily_qty = (
            trades.pivot_table(
                index="day", columns="ticker", values="signed_qty", aggfunc="sum"
            )
            .reindex(calendar)
            .fillna(0.0)
            .cumsum()
        )

        tickers = daily_qty.columns.tolist()
        if price_data is None:
            price_data = _fetch_price_history(tickers, start, end)
        prices = (
            price_data.reindex(columns=tickers)
            .reindex(calendar)
            .ffill()
            .bfill()
        )
        holdings_value = (daily_qty * prices).sum(axis=1)

    # Cassa: flusso netto cumulato di tutte le transazioni.
    cash_flow = (
        tx.assign(day=tx["date"].dt.normalize())
        .groupby("day")["amount_eur"]
        .sum()
        .reindex(calendar)
        .fillna(0.0)
        .cumsum()
    )

    result = pd.DataFrame(
        {
            "holdings_value": holdings_value,
            "cash": cash_flow,
        },
        index=calendar,
    )
    result["total_value"] = result["holdings_value"] + result["cash"]
    result.index.name = "date"
    return result


# --------------------------------------------------------------------------- #
# Rendimenti: TWR e MWR (XIRR)
# --------------------------------------------------------------------------- #
def _xirr(cashflows: list[tuple[pd.Timestamp, float]], *, guess: float = 0.1) -> float:
    """Calcola l'XIRR (tasso interno di rendimento money-weighted).

    Risolve ``sum(cf / (1+r)**years) = 0`` con Newton, fallback su Brent.

    Parameters
    ----------
    cashflows:
        Lista di ``(data, importo)``. Importi negativi = esborsi, positivi =
        incassi (incluso il valore finale del portafoglio).
    guess:
        Stima iniziale del tasso annuo.
    """
    if len(cashflows) < 2:
        return float("nan")
    cashflows = sorted(cashflows, key=lambda c: c[0])
    t0 = cashflows[0][0]
    years = np.array([(d - t0).days / 365.0 for d, _ in cashflows])
    amounts = np.array([a for _, a in cashflows], dtype="float64")

    if not (amounts.min() < 0 < amounts.max()):
        return float("nan")  # servono flussi di segno opposto

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    from scipy.optimize import brentq, newton

    try:
        return float(newton(npv, guess, maxiter=100, tol=1e-8))
    except (RuntimeError, OverflowError, FloatingPointError):
        try:
            return float(brentq(npv, -0.9999, 10.0, maxiter=200))
        except ValueError:
            logger.warning("XIRR non convergente")
            return float("nan")


def total_return(
    transactions_df: pd.DataFrame,
    *,
    price_data: pd.DataFrame | None = None,
    current_prices: dict[str, float] | None = None,
    end: pd.Timestamp | date | None = None,
) -> dict[str, float]:
    """Rendimento complessivo come TWR e MWR (XIRR).

    * **TWR** (Time-Weighted Return): concatena i rendimenti giornalieri del
      valore del portafoglio, neutralizzando l'effetto e il timing dei flussi
      di capitale. È la misura corretta per valutare la *strategia*.
    * **MWR** (Money-Weighted Return / XIRR): tasso che azzera il NPV di tutti i
      flussi di cassa più il valore finale del portafoglio. Riflette anche il
      *timing* degli investimenti dell'utente.

    Parameters
    ----------
    transactions_df:
        DataFrame canonico.
    price_data, current_prices, end:
        Override opzionali per evitare la rete (test/offline).

    Returns
    -------
    dict
        ``{"twr": float, "mwr": float, "final_value": float}`` (valori annui per
        l'MWR, cumulato per il TWR).
    """
    pv = portfolio_value_over_time(transactions_df, price_data=price_data, end=end)
    if pv.empty:
        return {"twr": float("nan"), "mwr": float("nan"), "final_value": 0.0}

    # --- TWR -------------------------------------------------------------- #
    # Flusso netto giornaliero verso le posizioni: acquisti (cash speso) meno
    # ricavi da vendite. I dividendi sono trattati come rendimento, non flusso.
    tx = transactions_df.copy()
    tx["day"] = tx["date"].dt.normalize()
    buy_cost = (
        tx[tx["type"].eq("buy")].groupby("day")["amount_eur"].sum().abs()
    )
    sell_proceeds = tx[tx["type"].eq("sell")].groupby("day")["amount_eur"].sum()
    flow = (
        buy_cost.subtract(sell_proceeds, fill_value=0.0)
        .reindex(pv.index)
        .fillna(0.0)
    )

    v = pv["holdings_value"]
    v_prev = v.shift(1)
    # r_t = (V_t - F_t) / V_{t-1} - 1 ; valido solo dove c'è capitale investito.
    daily_ret = (v - flow) / v_prev - 1.0
    daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    twr = float((1.0 + daily_ret).prod() - 1.0)

    # --- MWR / XIRR ------------------------------------------------------- #
    cashflows: list[tuple[pd.Timestamp, float]] = [
        (row.date.normalize(), float(row.amount_eur))
        for row in tx.itertuples(index=False)
    ]
    unreal = unrealized_pnl(transactions_df, current_prices=current_prices)
    final_value = float(unreal["market_value_eur"].sum()) if not unreal.empty else 0.0
    if final_value and not np.isnan(final_value):
        cashflows.append((pv.index[-1], final_value))
    mwr = _xirr(cashflows)

    return {"twr": twr, "mwr": mwr, "final_value": final_value}


# --------------------------------------------------------------------------- #
# Metriche di rischio
# --------------------------------------------------------------------------- #
def max_drawdown(portfolio_value_series: pd.Series) -> float:
    """Massimo drawdown (perdita massima dal picco precedente) come frazione.

    Returns un valore <= 0 (es. ``-0.25`` = -25%). ``0.0`` se la serie non
    registra cali.
    """
    series = pd.Series(portfolio_value_series).dropna().astype("float64")
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max.replace(0, np.nan)
    result = drawdown.min()
    return float(result) if pd.notna(result) else 0.0


def sharpe_ratio(
    portfolio_value_series: pd.Series,
    risk_free_rate: float = 0.03,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sharpe ratio annualizzato a partire da una serie di valori.

    Calcola i rendimenti periodali dalla serie, sottrae il risk-free per
    periodo e annualizza media e deviazione standard.

    Parameters
    ----------
    portfolio_value_series:
        Serie del valore del portafoglio (idealmente giornaliera).
    risk_free_rate:
        Tasso privo di rischio annuo (default 3%).
    periods_per_year:
        Periodi per anno per l'annualizzazione (252 per dati giornalieri).

    Returns
    -------
    float
        Sharpe annualizzato; ``nan`` se la volatilità è nulla o i dati sono
        insufficienti.
    """
    series = pd.Series(portfolio_value_series).dropna().astype("float64")
    returns = series.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


# --------------------------------------------------------------------------- #
# Allocazione
# --------------------------------------------------------------------------- #
def _infer_asset_class(ticker: str) -> str:
    """Euristica leggera per dedurre l'asset class dal ticker/suffisso.

    Volutamente semplice: in Fase 3/4 potrà essere sostituita da un lookup su
    metadati di mercato.
    """
    t = ticker.upper()
    if t == "CASH":
        return "Cash"
    if any(t.endswith(sfx) for sfx in (".DE", ".MI", ".AS", ".PA")) or t.startswith("V"):
        return "ETF"
    if "-" in t or t.endswith("USD") or t in {"BTC", "ETH"}:
        return "Crypto"
    return "Equity"


def allocation_breakdown(
    transactions_df: pd.DataFrame,
    *,
    current_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Allocazione percentuale attuale per ticker e asset class.

    Usa il valore di mercato delle posizioni aperte (via :func:`unrealized_pnl`).

    Parameters
    ----------
    transactions_df:
        DataFrame canonico.
    current_prices:
        Mappa prezzi opzionale per evitare la rete (test).

    Returns
    -------
    pd.DataFrame
        Indicizzato per ``ticker`` con ``market_value_eur``, ``asset_class`` e
        ``weight_pct`` (somma 100). Ordinato per peso decrescente.
    """
    unreal = unrealized_pnl(transactions_df, current_prices=current_prices)
    if unreal.empty:
        return pd.DataFrame(
            columns=["market_value_eur", "asset_class", "weight_pct"]
        ).rename_axis("ticker")

    df = unreal[["market_value_eur"]].copy()
    df["asset_class"] = [_infer_asset_class(t) for t in df.index]
    total = df["market_value_eur"].sum()
    df["weight_pct"] = (
        100.0 * df["market_value_eur"] / total if total else np.nan
    )
    return df.sort_values("weight_pct", ascending=False)
