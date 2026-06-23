"""Backtest: motore di simulazione e strategie (Fase 2)."""

from src.backtest.engine import (
    BacktestResult,
    backtest_strategy,
    compare_strategies,
    run_backtest,
)
from src.backtest.strategies import (
    BuyAndHold,
    Momentum,
    SMACrossover,
    Strategy,
)

__all__ = [
    "BacktestResult",
    "run_backtest",
    "backtest_strategy",
    "compare_strategies",
    "Strategy",
    "BuyAndHold",
    "SMACrossover",
    "Momentum",
]
