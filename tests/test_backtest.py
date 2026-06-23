"""Test unitari per il motore di backtest (Fase 2), deterministici e offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import backtest_strategy, compare_strategies, run_backtest
from src.backtest.strategies import BuyAndHold, Momentum, SMACrossover


@pytest.fixture
def prices():
    """Prezzi sintetici multi-asset deterministici."""
    from src.ingestion.sample_data import make_backtest_prices

    return make_backtest_prices(days=400)


@pytest.fixture
def single_asset_prices():
    """Singolo asset con crescita geometrica costante (+0,5%/giorno)."""
    idx = pd.date_range("2022-01-03", periods=100, freq="B")
    series = 100.0 * (1.005 ** np.arange(100))
    return pd.DataFrame({"X": series}, index=idx)


# --------------------------------------------------------------------------- #
# Correttezza del motore
# --------------------------------------------------------------------------- #
def test_buy_and_hold_matches_price_growth(single_asset_prices):
    """Buy&Hold full-invested su 1 asset deve replicare la crescita del prezzo."""
    res = backtest_strategy(single_asset_prices, BuyAndHold())
    px = single_asset_prices["X"]
    expected_total = px.iloc[-1] / px.iloc[0] - 1.0
    assert res.summary()["total_return"] == pytest.approx(expected_total, rel=1e-9)


def test_no_lookahead_first_day_flat(single_asset_prices):
    """Il primo giorno il capitale è invariato (pesi di ieri = 0)."""
    res = backtest_strategy(single_asset_prices, BuyAndHold())
    assert res.equity.iloc[0] == pytest.approx(res.initial_capital)


def test_fees_reduce_returns(prices):
    """Con costi di transazione il rendimento finale è inferiore."""
    strat = SMACrossover(fast=10, slow=30)
    no_fee = backtest_strategy(prices, strat, fee_bps=0.0)
    with_fee = backtest_strategy(prices, strat, fee_bps=50.0)
    assert with_fee.equity.iloc[-1] < no_fee.equity.iloc[-1]


def test_weights_aligned_and_bounded(prices):
    """I pesi applicati sono allineati ai prezzi e non negativi."""
    res = backtest_strategy(prices, SMACrossover(fast=10, slow=30))
    assert list(res.weights.index) == list(prices.index)
    assert (res.weights.values >= 0).all()
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-9).all()


# --------------------------------------------------------------------------- #
# Strategie
# --------------------------------------------------------------------------- #
def test_sma_crossover_invalid_params():
    with pytest.raises(ValueError):
        SMACrossover(fast=50, slow=20)  # fast >= slow


def test_sma_warmup_is_flat(prices):
    """Durante il warmup (prima di `slow` giorni) i pesi sono 0."""
    res = backtest_strategy(prices, SMACrossover(fast=10, slow=30))
    assert res.weights.iloc[:29].sum().sum() == pytest.approx(0.0)


def test_momentum_runs(prices):
    res = backtest_strategy(prices, Momentum(lookback=30))
    summ = res.summary()
    assert set(summ) == {
        "total_return", "cagr", "sharpe",
        "max_drawdown", "volatility", "n_rebalances",
    }
    assert np.isfinite(summ["total_return"])


# --------------------------------------------------------------------------- #
# Confronto strategie
# --------------------------------------------------------------------------- #
def test_compare_strategies_table(prices):
    strategies = [BuyAndHold(), SMACrossover(20, 50), Momentum(60)]
    results, table = compare_strategies(prices, strategies)
    assert len(results) == 3
    assert table.shape[0] == 3
    assert "sharpe" in table.columns
    assert "Buy & Hold" in table.index


def test_run_backtest_direct_weights(single_asset_prices):
    """API a basso livello: passare pesi espliciti."""
    weights = pd.DataFrame(1.0, index=single_asset_prices.index, columns=["X"])
    res = run_backtest(single_asset_prices, weights, initial_capital=5000.0)
    assert res.initial_capital == 5000.0
    assert res.equity.iloc[-1] > res.equity.iloc[0]
