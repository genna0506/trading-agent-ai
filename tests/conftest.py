"""Configurazione pytest: rende importabile il package `src` dalla root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Aggiunge la root del progetto al path così `import src...` funziona senza
# installare il package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.sample_data import (  # noqa: E402
    make_sample_price_history,
    make_sample_transactions,
)


@pytest.fixture
def sample_tx():
    """DataFrame canonico di transazioni di esempio."""
    return make_sample_transactions()


@pytest.fixture
def sample_prices(sample_tx):
    """Storico prezzi sintetico coerente con `sample_tx`."""
    return make_sample_price_history(sample_tx)


@pytest.fixture
def current_prices():
    """Prezzi correnti fissi (EUR) per test deterministici dell'unrealized."""
    return {"VWCE.DE": 130.0, "AAPL": 200.0, "ENI.MI": 15.0}
