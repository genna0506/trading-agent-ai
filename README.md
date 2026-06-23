# trading-ai

Toolkit modulare e production-grade per l'**analisi del trading**, sviluppato in 4 fasi
progressive. Questa repo contiene la **Fase 1** completa (ingestion + analisi di
portfolio) con l'architettura già predisposta per le fasi successive.

| Fase | Contenuto | Stato |
|------|-----------|-------|
| **1** | Ingestion dati (Fineco/Scalable) + metriche e grafici di portfolio | ✅ Completa |
| **2** | Motore di backtest delle strategie (`src/backtest/`) | ✅ Completa |
| 3 | Modelli ML (`src/ml/`) | 🟡 Stub |
| 4 | Agente AI con Claude (`src/agent/`) + dashboard Streamlit (`dashboard/`) | 🟡 Stub |

## Struttura

```
trading-ai/
├── data/
│   ├── raw/          # CSV esportati dai broker (Fineco, Scalable Capital)
│   └── processed/    # dataframe puliti salvati come parquet
├── notebooks/        # Jupyter notebook per l'esplorazione
├── src/
│   ├── config.py     # path del progetto (pathlib) e schema canonico
│   ├── ingestion/    # caricamento e pulizia dati  (loader.py, sample_data.py)
│   ├── analysis/     # metriche (metrics.py) e visualizzazioni (charts.py)
│   ├── backtest/     # Fase 2
│   ├── ml/           # Fase 3
│   └── agent/        # Fase 4 — tool dell'agente Claude
├── dashboard/        # Fase 4 — app Streamlit
├── tests/            # test unitari (pytest)
├── requirements.txt
└── README.md
```

## Requisiti

- **Python 3.11+**
- Librerie: pandas, numpy, yfinance, plotly, pyarrow, scipy, jupyter (vedi `requirements.txt`)

## 1. Setup dell'ambiente

```bash
cd trading-ai

# Crea e attiva un virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Caricare i dati reali

1. Esporta i movimenti dal tuo broker in CSV e copiali in `data/raw/`
   (es. `data/raw/fineco_2024.csv`, `data/raw/scalable_2024.csv`).
2. Il formato del broker viene **riconosciuto automaticamente** dalle intestazioni:
   - **Fineco**: `Data, Descrizione, Dare, Avere, Divisa, Saldo` (separatore `;`, numeri in formato IT).
   - **Scalable Capital**: colonne con `type, isin, shares, price, amount, currency, ...`.
3. Carica, normalizza e salva:

```python
from src.ingestion import loader

tx = loader.load_transactions()      # legge tutti i CSV in data/raw/
loader.save_processed(tx)            # -> data/processed/transactions.parquet
```

Tutte le transazioni vengono normalizzate in uno **schema unificato**:

| colonna | tipo | descrizione |
|---------|------|-------------|
| `date` | datetime | data dell'operazione |
| `ticker` | str | simbolo (es. `AAPL`, `VWCE.DE`) |
| `type` | str | `buy` / `sell` / `dividend` / `fee` / `tax` |
| `quantity` | float | quantità |
| `price` | float | prezzo unitario (valuta nativa) |
| `amount_eur` | float | importo in EUR (negativo = uscita) |
| `currency` | str | valuta nativa |
| `fx_rate` | float | tasso valuta→EUR (storico, da yfinance) |
| `broker` | str | broker di origine |

La conversione valutaria (EUR/USD, ecc.) usa i **tassi FX storici** scaricati da yfinance
con un merge *as-of* (ultimo tasso noto alla data dell'operazione).

## 3. Eseguire il notebook

```bash
# con il venv attivo
pip install jupyter           # già in requirements.txt
jupyter notebook notebooks/01_portfolio_analysis.ipynb
```

Il notebook gira **out-of-the-box** con dati di esempio sintetici (nessuna connessione
richiesta). Per usare i dati reali, sostituisci la cella di caricamento con
`loader.load_transactions()` e rimuovi gli override `price_data` / `current_prices`
(così i prezzi vengono scaricati da yfinance).

## 4. Metriche disponibili (`src/analysis/metrics.py`)

| Funzione | Output |
|----------|--------|
| `portfolio_value_over_time(tx)` | valore giornaliero (`holdings_value`, `cash`, `total_value`) |
| `realized_pnl(tx)` | P&L realizzato per ticker (**FIFO**, gestisce vendite parziali multi-lotto) |
| `unrealized_pnl(tx)` | P&L non realizzato mark-to-market (prezzi via yfinance) |
| `total_return(tx)` | **TWR** (time-weighted) e **MWR/XIRR** (money-weighted) |
| `max_drawdown(series)` | massimo drawdown |
| `sharpe_ratio(series)` | Sharpe annualizzato |
| `allocation_breakdown(tx)` | allocazione % per ticker e asset class |

> Per le metriche di rischio usa `holdings_value`: `total_value` include la cassa che,
> non essendo tracciati i versamenti negli export di soli trade, può risultare negativa.

Grafici Plotly in `src/analysis/charts.py`: `plot_portfolio_value`, `plot_allocation_pie`,
`plot_pnl_waterfall`, `plot_monthly_returns_heatmap`.

## 5. Backtest delle strategie (Fase 2)

Il motore in `src/backtest/` testa strategie di investimento sui prezzi storici.
Una **strategia** trasforma i prezzi in *pesi target*; il **motore** simula il
portafoglio (vettorizzato, senza look-ahead, con costi di transazione) e produce
la *equity curve*. Le metriche di rischio riusano la Fase 1.

```python
from src.ingestion.sample_data import make_backtest_prices
from src.backtest import BuyAndHold, SMACrossover, Momentum, compare_strategies
from src.backtest.plots import plot_strategy_comparison

prices = make_backtest_prices(days=750)              # o i tuoi prezzi reali
results, table = compare_strategies(
    prices, [BuyAndHold(), SMACrossover(20, 50), Momentum(60)], fee_bps=10
)
print(table)                                          # tabella di confronto
plot_strategy_comparison(results).show()              # grafico equity curve
```

Strategie incluse: `BuyAndHold` (benchmark), `SMACrossover(fast, slow)`
(incrocio medie mobili), `Momentum(lookback)`. Aggiungerne di nuove = creare una
classe con `generate_weights(prices)` in `src/backtest/strategies.py`.

Notebook dimostrativo: `notebooks/02_backtest_analysis.ipynb`.

## 6. Test

```bash
pip install pytest
pytest -q
```

I test girano **completamente offline** (i prezzi sono iniettati come parametri, niente rete).

## Note di design

- **Nessun path hardcoded**: tutti i percorsi derivano da `PROJECT_ROOT` in `src/config.py` (`pathlib`).
- **Vettorizzazione**: niente `iterrows()`; il FIFO usa `itertuples` per performance.
- **Architettura per la Fase 4**: le funzioni di `src/` sono pure e richiamabili direttamente
  come *tool* dell'agente Claude (vedi `src/agent/tools.py`).
- **Testabilità**: ogni funzione che tocca la rete accetta un override dei dati
  (`price_data` / `current_prices`).
