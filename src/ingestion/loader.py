"""Loader CSV flessibile per export dei broker (Fineco, Scalable Capital).

L'obiettivo del modulo è trasformare gli export eterogenei dei broker in un
unico DataFrame "canonico" di transazioni, con lo schema definito in
:data:`src.config.TRANSACTION_COLUMNS`.

Flusso tipico::

    from src.ingestion import loader
    tx = loader.load_transactions()              # legge tutti i CSV in data/raw
    path = loader.save_processed(tx)             # salva in data/processed/*.parquet

Note di design
--------------
* Il riconoscimento del formato (Fineco vs Scalable) è automatico, basato sulle
  intestazioni di colonna. È comunque possibile forzarlo passando ``broker=``.
* La conversione valutaria avviene una sola volta, in modo vettorizzato, sfruttando
  una cache dei tassi FX storici scaricati da yfinance (vedi :func:`get_fx_rates`).
* Nessun path è hardcoded: tutto deriva da :mod:`src.config`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    RAW_DIR,
    TRANSACTION_COLUMNS,
    VALID_TRANSACTION_TYPES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

# Mappa parole chiave (lower-case) -> tipo canonico, usata per classificare le
# descrizioni testuali degli export Fineco.
_FINECO_TYPE_KEYWORDS: dict[str, str] = {
    "acquisto": "buy",
    "sottoscrizione": "buy",
    "vendita": "sell",
    "rimborso": "sell",
    "dividendo": "dividend",
    "cedola": "dividend",
    "commission": "fee",
    "commissione": "fee",
    "imposta": "tax",
    "bollo": "tax",
    "ritenuta": "tax",
    "tobin": "tax",
}


# --------------------------------------------------------------------------- #
# FX helpers
# --------------------------------------------------------------------------- #
def get_fx_rates(
    currencies: list[str],
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> dict[str, pd.Series]:
    """Scarica i tassi di cambio storici *valuta -> EUR* da yfinance.

    Per ogni valuta diversa da EUR viene scaricata la coppia ``EUR<CUR>=X``
    (es. ``EURUSD=X``) e invertita per ottenere ``CUR -> EUR``. La serie è
    indicizzata per data (giornaliera) ed è forward-filled per coprire i giorni
    non di mercato.

    Parameters
    ----------
    currencies:
        Lista di codici valuta ISO (es. ``["USD", "GBP"]``). ``EUR`` viene
        ignorato.
    start, end:
        Intervallo temporale richiesto.

    Returns
    -------
    dict[str, pd.Series]
        Mappa ``{valuta: serie_tassi}`` dove ogni serie converte 1 unità della
        valuta in EUR. ``EUR`` mappa sempre su una serie costante pari a 1.0.

    Notes
    -----
    L'import di ``yfinance`` è lazy così il modulo resta importabile (e testabile)
    anche in ambienti senza rete o senza la dipendenza installata.
    """
    rates: dict[str, pd.Series] = {}
    needed = sorted({c.upper() for c in currencies if c and c.upper() != "EUR"})
    if not needed:
        return rates

    import yfinance as yf  # import lazy

    for cur in needed:
        pair = f"EUR{cur}=X"
        try:
            data = yf.download(
                pair,
                start=pd.Timestamp(start).normalize(),
                end=pd.Timestamp(end).normalize() + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=False,
            )
            if data.empty:
                raise ValueError("nessun dato restituito")
            close = data["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance multi-col
                close = close.iloc[:, 0]
            # EUR<CUR>=X = quante unità di CUR per 1 EUR -> invertiamo per CUR->EUR
            rates[cur] = (1.0 / close).rename(cur)
        except Exception as exc:  # noqa: BLE001 - vogliamo degradare con grazia
            logger.warning("FX non disponibile per %s (%s); uso 1.0", cur, exc)
            idx = pd.date_range(start, end, freq="D")
            rates[cur] = pd.Series(1.0, index=idx, name=cur)
    return rates


def _apply_fx(df: pd.DataFrame) -> pd.DataFrame:
    """Popola ``fx_rate`` e ``amount_eur`` per le righe non in EUR.

    Le righe già in EUR ottengono ``fx_rate = 1.0``. Per le altre valute si
    effettua un merge "as-of" (ultimo tasso noto <= data operazione) per evitare
    qualsiasi loop riga-per-riga.
    """
    df = df.copy()
    df["currency"] = df["currency"].fillna("EUR").str.upper()
    df["fx_rate"] = np.where(df["currency"].eq("EUR"), 1.0, np.nan)

    non_eur = df.loc[df["currency"].ne("EUR")]
    if not non_eur.empty:
        fx = get_fx_rates(
            non_eur["currency"].unique().tolist(),
            non_eur["date"].min(),
            non_eur["date"].max(),
        )
        for cur, series in fx.items():
            series = series.sort_index()
            mask = df["currency"].eq(cur)
            sub = df.loc[mask, ["date"]].sort_values("date")
            merged = pd.merge_asof(
                sub,
                series.rename("rate").reset_index().rename(columns={"index": "date"}),
                on="date",
                direction="backward",
            )
            df.loc[sub.index, "fx_rate"] = merged["rate"].to_numpy()

    df["fx_rate"] = df["fx_rate"].fillna(1.0)

    # amount_eur: se mancante, lo deriviamo da quantità*prezzo*fx. Se già presente
    # (Fineco fornisce importi in EUR) lo lasciamo invariato.
    derived = df["quantity"].fillna(0) * df["price"].fillna(0) * df["fx_rate"]
    df["amount_eur"] = df["amount_eur"].where(df["amount_eur"].notna(), derived)
    return df


# --------------------------------------------------------------------------- #
# Riconoscimento formato
# --------------------------------------------------------------------------- #
def _detect_broker(columns: list[str]) -> str:
    """Indovina il broker dalle intestazioni di colonna.

    Returns ``"fineco"`` o ``"scalable"``; solleva ``ValueError`` se nessuno
    dei due pattern combacia.
    """
    lower = {c.strip().lower() for c in columns}
    fineco_markers = {"dare", "avere", "saldo"}
    scalable_markers = {"shares", "isin"} | {"price", "fee"}
    if fineco_markers & lower:
        return "fineco"
    if "isin" in lower or "shares" in lower or {"type", "amount"} <= lower:
        return "scalable"
    raise ValueError(
        f"Formato broker non riconosciuto dalle colonne: {sorted(lower)}"
    )


# --------------------------------------------------------------------------- #
# Parser Fineco
# --------------------------------------------------------------------------- #
def _classify_fineco_row(description: str) -> str:
    """Mappa una descrizione testuale Fineco su un tipo canonico."""
    text = (description or "").lower()
    for keyword, kind in _FINECO_TYPE_KEYWORDS.items():
        if keyword in text:
            return kind
    return "fee"  # fallback prudente per movimenti non classificati


def load_fineco_csv(path: str | Path, *, sep: str = ";") -> pd.DataFrame:
    """Carica un export Fineco e lo normalizza nello schema canonico.

    Colonne attese (case-insensitive): ``Data, Descrizione, Dare, Avere,
    Divisa, Saldo``. I numeri usano la virgola decimale e il punto come
    separatore delle migliaia (formato italiano).

    Parameters
    ----------
    path:
        Percorso del file CSV.
    sep:
        Separatore di campo (Fineco usa ``;``).

    Returns
    -------
    pd.DataFrame
        DataFrame con le colonne di :data:`TRANSACTION_COLUMNS` (``fx_rate`` e
        ``amount_eur`` finalizzati a valle da :func:`_apply_fx`).
    """
    raw = pd.read_csv(path, sep=sep, dtype=str).rename(columns=lambda c: c.strip())
    cols = {c.lower(): c for c in raw.columns}

    def col(name: str) -> pd.Series:
        return raw[cols[name]] if name in cols else pd.Series([None] * len(raw))

    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(col("data"), dayfirst=True, errors="coerce")
    description = col("descrizione").fillna("")
    out["type"] = description.map(_classify_fineco_row)

    # Ticker: tentiamo di estrarlo dalla descrizione (ultima parola in maiuscolo
    # o tra parentesi). Se assente resta None e verrà gestito a valle.
    out["ticker"] = (
        description.str.extract(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b", expand=False)
    )

    dare = _to_float_it(col("dare"))   # uscite (negativo per noi)
    avere = _to_float_it(col("avere"))  # entrate
    out["amount_eur"] = avere.fillna(0) - dare.fillna(0)

    out["currency"] = col("divisa").fillna("EUR").str.upper()
    out["quantity"] = np.nan  # Fineco non espone sempre la quantità nel movimento
    out["price"] = np.nan
    out["fx_rate"] = np.nan
    out["broker"] = "fineco"
    return out


# --------------------------------------------------------------------------- #
# Parser Scalable Capital
# --------------------------------------------------------------------------- #
def load_scalable_csv(path: str | Path, *, sep: str = ";") -> pd.DataFrame:
    """Carica un export Scalable Capital e lo normalizza nello schema canonico.

    Scalable separa trade, dividendi e commissioni; le colonne tipiche sono
    ``date, type, isin, description, shares, price, amount, currency, fee, tax``.
    Il parser è tollerante: usa solo le colonne presenti.

    Parameters
    ----------
    path:
        Percorso del file CSV.
    sep:
        Separatore di campo.

    Returns
    -------
    pd.DataFrame
        DataFrame con le colonne canoniche (FX finalizzato a valle).
    """
    raw = pd.read_csv(path, sep=sep, dtype=str).rename(columns=lambda c: c.strip())
    cols = {c.lower(): c for c in raw.columns}

    def col(name: str) -> pd.Series:
        return raw[cols[name]] if name in cols else pd.Series([None] * len(raw))

    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(col("date"), errors="coerce", dayfirst=True)

    raw_type = col("type").fillna("").str.lower()
    type_map = {
        "buy": "buy",
        "purchase": "buy",
        "sell": "sell",
        "sale": "sell",
        "dividend": "dividend",
        "distribution": "dividend",
        "fee": "fee",
        "interest": "dividend",
        "tax": "tax",
    }
    out["type"] = raw_type.map(type_map).fillna("fee")

    # Scalable usa l'ISIN come identificativo; lo manteniamo come ticker se non
    # è presente una colonna ticker dedicata.
    out["ticker"] = col("ticker").where(col("ticker").notna(), col("isin"))
    out["quantity"] = _to_float_en(col("shares"))
    out["price"] = _to_float_en(col("price"))
    out["amount_eur"] = _to_float_en(col("amount"))
    out["currency"] = col("currency").fillna("EUR").str.upper()
    out["fx_rate"] = np.nan
    out["broker"] = "scalable"
    return out


# --------------------------------------------------------------------------- #
# Numeric parsing helpers
# --------------------------------------------------------------------------- #
def _to_float_it(series: pd.Series) -> pd.Series:
    """Converte stringhe numeriche in formato italiano (1.234,56) in float."""
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = (
        series.astype("string")
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _to_float_en(series: pd.Series) -> pd.Series:
    """Converte stringhe numeriche in formato anglosassone (1,234.56) in float."""
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# --------------------------------------------------------------------------- #
# Validazione & API pubblica
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Finalizza un DataFrame grezzo: FX, ordinamento, tipi e validazione.

    Garantisce che siano presenti esattamente le :data:`TRANSACTION_COLUMNS`,
    nell'ordine corretto e con i dtype attesi.
    """
    df = _apply_fx(df)

    # Riempiamo i ticker mancanti con un segnaposto coerente (es. movimenti di
    # solo cassa) così le aggregazioni per ticker non perdono righe.
    df["ticker"] = df["ticker"].fillna("CASH").astype(str)
    df["type"] = df["type"].astype(str).str.lower()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("float64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    df["amount_eur"] = pd.to_numeric(df["amount_eur"], errors="coerce").astype("float64")
    df["fx_rate"] = pd.to_numeric(df["fx_rate"], errors="coerce").fillna(1.0)

    invalid = set(df["type"].unique()) - VALID_TRANSACTION_TYPES
    if invalid:
        logger.warning("Tipi transazione non riconosciuti rimappati a 'fee': %s", invalid)
        df.loc[df["type"].isin(invalid), "type"] = "fee"

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[list(TRANSACTION_COLUMNS)]


def load_csv(path: str | Path, *, broker: str | None = None) -> pd.DataFrame:
    """Carica un singolo CSV rilevando (o forzando) il formato del broker.

    Parameters
    ----------
    path:
        Percorso del file.
    broker:
        ``"fineco"`` o ``"scalable"`` per forzare il parser; ``None`` per
        rilevamento automatico.
    """
    path = Path(path)
    header = pd.read_csv(path, sep=None, engine="python", nrows=0)
    chosen = broker or _detect_broker(list(header.columns))
    parser = {"fineco": load_fineco_csv, "scalable": load_scalable_csv}[chosen]
    logger.info("Carico %s come formato '%s'", path.name, chosen)
    return parser(path)


def load_transactions(
    raw_dir: str | Path = RAW_DIR,
    *,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Carica e unifica tutti i CSV presenti in ``raw_dir``.

    Ogni file viene parsato col formato rilevato automaticamente, quindi i
    risultati sono concatenati e normalizzati in un unico DataFrame canonico.

    Parameters
    ----------
    raw_dir:
        Cartella dei CSV grezzi (default ``data/raw``).
    pattern:
        Glob dei file da includere.

    Returns
    -------
    pd.DataFrame
        Transazioni unificate con lo schema :data:`TRANSACTION_COLUMNS`. Vuoto
        (ma con le colonne corrette) se non ci sono file.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(pattern))
    if not files:
        logger.warning("Nessun CSV trovato in %s", raw_dir)
        return pd.DataFrame(columns=list(TRANSACTION_COLUMNS))

    frames = [load_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    return normalize(combined)


def save_processed(
    df: pd.DataFrame,
    *,
    filename: str = "transactions.parquet",
) -> Path:
    """Salva il DataFrame normalizzato in ``data/processed`` come parquet.

    Returns il path del file scritto.
    """
    ensure_dirs()
    out_path = PROCESSED_DIR / filename
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Salvate %d transazioni in %s", len(df), out_path)
    return out_path


def load_processed(filename: str = "transactions.parquet") -> pd.DataFrame:
    """Rilegge le transazioni salvate da ``data/processed``."""
    return pd.read_parquet(PROCESSED_DIR / filename, engine="pyarrow")
