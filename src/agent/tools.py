"""Tool dell'agente AI (Claude) — STUB Fase 4.

Mostra come le funzioni di :mod:`src.analysis.metrics` verranno esposte come
*tool* all'agente Claude (Fase 4). Le funzioni di analisi NON dipendono
dall'agente: l'agente è un sottile strato di orchestrazione sopra `src/`, in
linea con l'architettura modulare.

Quando si implementerà la Fase 4, ogni tool sarà una funzione pura che:
1. riceve argomenti JSON-serializzabili dall'agente,
2. carica le transazioni (parquet) e chiama la metrica corrispondente,
3. restituisce un risultato JSON-serializzabile.

Esempio di schema tool per l'API Claude (riferimento)::

    {
        "name": "get_realized_pnl",
        "description": "P&L realizzato per ticker con metodo FIFO.",
        "input_schema": {"type": "object", "properties": {}},
    }
"""

from __future__ import annotations

from typing import Any

from src.analysis import metrics
from src.ingestion.loader import load_processed


def get_realized_pnl() -> list[dict[str, Any]]:
    """Tool: P&L realizzato per ticker (FIFO). Ritorna record JSON-serializzabili."""
    tx = load_processed()
    return metrics.realized_pnl(tx).reset_index().to_dict(orient="records")


def get_allocation() -> list[dict[str, Any]]:
    """Tool: allocazione attuale per ticker e asset class."""
    tx = load_processed()
    return metrics.allocation_breakdown(tx).reset_index().to_dict(orient="records")


# Registro dei tool esposti all'agente (popolato nella Fase 4).
TOOL_REGISTRY = {
    "get_realized_pnl": get_realized_pnl,
    "get_allocation": get_allocation,
}
