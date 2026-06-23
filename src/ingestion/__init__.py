"""Ingestion: caricamento e normalizzazione dei dati dei broker (Fase 1)."""

from src.ingestion.loader import (
    load_transactions,
    load_fineco_csv,
    load_scalable_csv,
    save_processed,
)

__all__ = [
    "load_transactions",
    "load_fineco_csv",
    "load_scalable_csv",
    "save_processed",
]
