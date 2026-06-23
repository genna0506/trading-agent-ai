"""Costanti di configurazione e path del progetto.

Tutti i percorsi sono derivati da :data:`PROJECT_ROOT` usando ``pathlib`` in
modo da non avere mai path hardcoded sparsi nel codice. Importare i path da
qui (es. ``from src.config import PROCESSED_DIR``) invece di costruirli a mano.
"""

from __future__ import annotations

from pathlib import Path

# Root del progetto = cartella che contiene `src/` (due livelli sopra questo file:
# src/config.py -> src/ -> <root>).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Cartelle dati
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Altre cartelle utili
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Schema canonico di una transazione normalizzata. Usato dal loader e dalle
# metriche come unica fonte di verità sulle colonne attese.
TRANSACTION_COLUMNS: tuple[str, ...] = (
    "date",        # datetime64[ns] - data di regolamento dell'operazione
    "ticker",      # str            - simbolo (es. "AAPL", "VWCE.DE")
    "type",        # str            - buy | sell | dividend | fee | tax
    "quantity",    # float          - quantità (positiva; il segno è dato da `type`)
    "price",       # float          - prezzo unitario nella valuta nativa
    "amount_eur",  # float          - importo in EUR (negativo = uscita di cassa)
    "currency",    # str            - valuta nativa dell'operazione (es. "USD")
    "fx_rate",     # float          - tasso valuta->EUR usato per la conversione
    "broker",      # str            - "fineco" | "scalable" | ...
)

# Tipi di transazione ammessi.
VALID_TRANSACTION_TYPES: frozenset[str] = frozenset(
    {"buy", "sell", "dividend", "fee", "tax"}
)


def ensure_dirs() -> None:
    """Crea le cartelle dati se non esistono (idempotente)."""
    for directory in (RAW_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
