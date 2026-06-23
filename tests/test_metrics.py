"""Test unitari per src/analysis/metrics.py usando dati dummy deterministici.

I prezzi sono iniettati nelle funzioni (parametri ``current_prices`` /
``price_data``) così i test girano completamente offline, senza yfinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import metrics


# --------------------------------------------------------------------------- #
# FIFO realized P&L
# --------------------------------------------------------------------------- #
def test_realized_pnl_partial_sale_across_lots(sample_tx):
    """La vendita di 40 VWCE consuma il primo lotto (50@100): P&L = 40*(120-100)."""
    result = metrics.realized_pnl(sample_tx)
    assert "VWCE.DE" in result.index
    assert result.loc["VWCE.DE", "realized_pnl_eur"] == pytest.approx(800.0)


def test_realized_pnl_no_sales_means_empty_or_absent(sample_tx):
    """Ticker senza vendite non compaiono nel realized."""
    result = metrics.realized_pnl(sample_tx)
    assert "AAPL" not in result.index  # solo acquisti
    assert "ENI.MI" not in result.index


def test_fifo_multi_lot_full_consumption():
    """Vendita che attraversa due lotti deve sommare i P&L per lotto."""
    tx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "ticker": ["X", "X", "X"],
            "type": ["buy", "buy", "sell"],
            "quantity": [10.0, 10.0, 15.0],
            "price": [10.0, 20.0, 30.0],
            "amount_eur": [-100.0, -200.0, 450.0],
            "currency": ["EUR"] * 3,
            "fx_rate": [1.0] * 3,
            "broker": ["t"] * 3,
        }
    )
    # 10@10 (gain 10*20=200) + 5@20 (gain 5*10=50) = 250
    result = metrics.realized_pnl(tx)
    assert result.loc["X", "realized_pnl_eur"] == pytest.approx(250.0)


# --------------------------------------------------------------------------- #
# Holdings & unrealized
# --------------------------------------------------------------------------- #
def test_current_holdings(sample_tx):
    holdings = metrics.current_holdings(sample_tx)
    assert holdings.loc["VWCE.DE"] == pytest.approx(40.0)
    assert holdings.loc["AAPL"] == pytest.approx(30.0)
    assert holdings.loc["ENI.MI"] == pytest.approx(100.0)


def test_unrealized_pnl_uses_remaining_fifo_lots(sample_tx, current_prices):
    result = metrics.unrealized_pnl(sample_tx, current_prices=current_prices)
    # VWCE residuo: 10@100 + 30@110 = 4300 di costo; mv = 40*130 = 5200
    assert result.loc["VWCE.DE", "cost_basis_eur"] == pytest.approx(4300.0)
    assert result.loc["VWCE.DE", "market_value_eur"] == pytest.approx(5200.0)
    assert result.loc["VWCE.DE", "unrealized_pnl_eur"] == pytest.approx(900.0)
    # AAPL in USD convertito: 20*138 + 10*167.4 = 4434
    assert result.loc["AAPL", "cost_basis_eur"] == pytest.approx(4434.0)
    assert result.loc["AAPL", "unrealized_pnl_eur"] == pytest.approx(1566.0)


# --------------------------------------------------------------------------- #
# Rischio: drawdown & sharpe
# --------------------------------------------------------------------------- #
def test_max_drawdown_known_series():
    series = pd.Series([100, 120, 90, 110, 80])
    # peggior calo dal picco 120 a 80 -> -40/120
    assert metrics.max_drawdown(series) == pytest.approx(-1 / 3)


def test_max_drawdown_monotonic_increase_is_zero():
    series = pd.Series([100, 101, 102, 103])
    assert metrics.max_drawdown(series) == pytest.approx(0.0)


def test_sharpe_ratio_returns_float():
    rng = np.random.default_rng(0)
    series = pd.Series(1000 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)))
    sr = metrics.sharpe_ratio(series)
    assert isinstance(sr, float)
    assert np.isfinite(sr)


def test_sharpe_ratio_insufficient_data_is_nan():
    assert np.isnan(metrics.sharpe_ratio(pd.Series([100.0])))


def test_sharpe_ratio_zero_volatility_is_nan():
    # Serie piatta -> rendimenti tutti 0 -> volatilità esattamente nulla.
    flat = pd.Series([100.0] * 10)
    assert np.isnan(metrics.sharpe_ratio(flat, risk_free_rate=0.0))


# --------------------------------------------------------------------------- #
# Allocazione
# --------------------------------------------------------------------------- #
def test_allocation_weights_sum_to_100(sample_tx, current_prices):
    alloc = metrics.allocation_breakdown(sample_tx, current_prices=current_prices)
    assert alloc["weight_pct"].sum() == pytest.approx(100.0)
    assert "asset_class" in alloc.columns
    # AAPL ha il market value più alto (6000)
    assert alloc.index[0] == "AAPL"


# --------------------------------------------------------------------------- #
# Valore nel tempo & rendimenti
# --------------------------------------------------------------------------- #
def test_portfolio_value_over_time_shape(sample_tx, sample_prices):
    pv = metrics.portfolio_value_over_time(sample_tx, price_data=sample_prices)
    assert not pv.empty
    assert list(pv.columns) == ["holdings_value", "cash", "total_value"]
    assert pv.index.is_monotonic_increasing
    # total = holdings + cash
    np.testing.assert_allclose(
        pv["total_value"].values,
        (pv["holdings_value"] + pv["cash"]).values,
    )


def test_total_return_structure(sample_tx, sample_prices, current_prices):
    out = metrics.total_return(
        sample_tx, price_data=sample_prices, current_prices=current_prices
    )
    assert set(out) == {"twr", "mwr", "final_value"}
    # final_value = somma market value posizioni aperte = 5200+6000+1500
    assert out["final_value"] == pytest.approx(12700.0)
    assert isinstance(out["twr"], float)


def test_empty_transactions_safe():
    empty = pd.DataFrame(
        columns=[
            "date", "ticker", "type", "quantity", "price",
            "amount_eur", "currency", "fx_rate", "broker",
        ]
    )
    pv = metrics.portfolio_value_over_time(empty)
    assert pv.empty
    assert metrics.realized_pnl(empty).empty
