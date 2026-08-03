# -*- coding: utf-8 -*-
"""
fetcher_swing.py — Pipeline de datos para el Swing Scanner (independiente de DeepView)
=====================================================================================
Descarga automáticamente los componentes de S&P 500, Nasdaq-100, IBEX 35 y DAX 40,
baja ~300 sesiones de velas diarias con yfinance, calcula indicadores y detecta
setups (A: consolidación/VCP simplificado, B: pullback a EMA10/21) y exporta
swing_data.json para cargarlo en swing_scanner.html.

Uso:
    pip install yfinance pandas lxml
    python fetcher_swing.py

Salida:
    swing_data.json  (en el mismo directorio)

Notas:
- Sin tickers a mano: los índices se leen de Wikipedia (con listas de respaldo).
- Timeframe DIARIO (EMA10/21, SMA50/150/200, ADR 20 sesiones), no semanal.
- Earnings: mejor esfuerzo vía yfinance; si no hay dato, days_to_earnings = null.
"""

import json
import math
import sys
import time
from datetime import datetime, timezone

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Falta yfinance. Instala con: pip install yfinance pandas lxml")

# ----------------------------------------------------------------------------
# 1. UNIVERSO AUTOMÁTICO
# ----------------------------------------------------------------------------

WIKI = {
    "SP500":  ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
    "NDX":    ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    "IBEX35": ("https://es.wikipedia.org/wiki/IBEX_35", "Ticker"),
    "DAX40":  ("https://en.wikipedia.org/wiki/DAX", "Ticker"),
}

# Respaldo por si Wikipedia cambia el formato de la tabla
FALLBACK_IBEX35 = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC", "BBVA.MC",
    "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC", "ENG.MC", "FDR.MC",
    "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC", "IDR.MC", "ITX.MC", "LOG.MC",
    "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC", "PUIG.MC", "RED.MC", "REP.MC",
    "ROVI.MC", "SAB.MC", "SAN.MC", "SCYR.MC", "SLR.MC", "TEF.MC", "UNI.MC",
]


def _clean_symbol(sym: str, market: str) -> str:
    sym = str(sym).strip().upper()
    if market in ("SP500", "NDX"):
        return sym.replace(".", "-")          # BRK.B -> BRK-B (formato Yahoo)
    if market == "IBEX35":
        return sym if sym.endswith(".MC") else sym + ".MC"
    if market == "DAX40":
        return sym if sym.endswith(".DE") else sym + ".DE"
    return sym


def get_universe() -> dict:
    """Devuelve {ticker: market}. Un ticker en varios índices conserva el primero."""
    universe = {}
    for market, (url, col) in WIKI.items():
        symbols = []
        try:
            tables = pd.read_html(url)
            for t in tables:
                cols = [str(c) for c in t.columns]
                match = next((c for c in cols if col.lower() in c.lower()), None)
                if match and len(t) > 20:
                    symbols = [_clean_symbol(s, market) for s in t[match].dropna()]
                    break
        except Exception as e:
            print(f"[AVISO] Wikipedia falló para {market}: {e}")
        if not symbols and market == "IBEX35":
            symbols = FALLBACK_IBEX35
            print("[AVISO] Usando lista de respaldo para IBEX35")
        if not symbols:
            print(f"[AVISO] {market} sin componentes — se omite")
            continue
        for s in symbols:
            universe.setdefault(s, market)
        print(f"{market}: {len(symbols)} componentes")
    return universe


# ----------------------------------------------------------------------------
# 2. DESCARGA DE VELAS DIARIAS
# ----------------------------------------------------------------------------

def download_history(tickers, period="2y"):
    """Descarga en bloque. Devuelve dict ticker -> DataFrame OHLCV."""
    print(f"Descargando {len(tickers)} tickers…")
    data = yf.download(
        tickers=list(tickers), period=period, interval="1d",
        group_by="ticker", auto_adjust=True, threads=True, progress=True,
    )
    out = {}
    for t in tickers:
        try:
            df = data[t].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
            df = df.dropna(subset=["Close"])
            if len(df) >= 220:                # mínimo para SMA200 fiable
                out[t] = df
        except Exception:
            continue
    print(f"Con historial suficiente: {len(out)}")
    return out


# ----------------------------------------------------------------------------
# 3. INDICADORES Y SETUPS (todo sobre velas DIARIAS)
# ----------------------------------------------------------------------------

def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def compute_metrics(df: pd.DataFrame) -> dict | None:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    if len(c) < 220:
        return None
    last = float(c.iloc[-1])

    sma50, sma150, sma200 = sma(c, 50), sma(c, 150), sma(c, 200)
    ema10, ema21 = ema(c, 10), ema(c, 21)

    # SMA200 subiendo al menos 1 mes (21 sesiones)
    sma200_up = bool(sma200.iloc[-1] > sma200.iloc[-22])

    lo52 = float(l.iloc[-252:].min())
    hi52 = float(h.iloc[-252:].max())
    pct_over_low = (last / lo52 - 1) * 100
    pct_under_high = (last / hi52 - 1) * 100  # negativo o 0

    # Trend Template (Minervini, versión diaria)
    trend_template = bool(
        last > sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]
        and sma200_up
        and pct_over_low >= 30
        and pct_under_high >= -25
    )

    # ADR% medio 20 sesiones
    adr = float(((h / l - 1).iloc[-20:].mean()) * 100)

    # Volumen medio en divisa, 20 sesiones
    dollar_vol = float((c * v).iloc[-20:].mean())

    # Momentum previo: mejor subida en ventana de 40 sesiones dentro del último año
    roll_min = c.rolling(40).min()
    burst = float(((c / roll_min - 1).iloc[-252:].max()) * 100)

    # RS ponderado estilo IBD (percentil se calcula después sobre el universo)
    def ret(n):
        return float(c.iloc[-1] / c.iloc[-n] - 1) if len(c) > n else 0.0
    rs_raw = 0.4 * ret(63) + 0.2 * ret(126) + 0.2 * ret(189) + 0.2 * ret(252)

    # --- Setup A: consolidación con contracción (VCP simplificado) ---
    hi13w = float(h.iloc[-65:].max())
    rng_last10 = float(h.iloc[-10:].max() / l.iloc[-10:].min() - 1)
    rng_prev10 = float(h.iloc[-20:-10].max() / l.iloc[-20:-10].min() - 1)
    vol_drying = bool(v.iloc[-10:].mean() < v.iloc[-30:-10].mean())
    setup_a = bool(
        trend_template
        and last >= hi13w * 0.85                    # a menos del 15% del pivote
        and rng_prev10 > 0 and rng_last10 < rng_prev10  # contracción
        and vol_drying
    )
    pivot = round(hi13w, 2)

    # --- Setup B: pullback a EMA10/21 tras impulso ---
    made_20d_high_recently = bool((h.iloc[-15:] >= h.rolling(20).max().iloc[-15:]).any())
    touched_ema = bool((l.iloc[-3:] <= ema10.iloc[-3:] * 1.01).any()
                       or (l.iloc[-3:] <= ema21.iloc[-3:] * 1.01).any())
    above_ema21 = bool((c.iloc[-10:] > ema21.iloc[-10:] * 0.99).all())
    setup_b = bool(trend_template and made_20d_high_recently and touched_ema
                   and above_ema21 and vol_drying)

    return {
        "close": round(last, 2),
        "sma50": round(float(sma50.iloc[-1]), 2),
        "sma150": round(float(sma150.iloc[-1]), 2),
        "sma200": round(float(sma200.iloc[-1]), 2),
        "ema10": round(float(ema10.iloc[-1]), 2),
        "ema21": round(float(ema21.iloc[-1]), 2),
        "trend_template": trend_template,
        "pct_over_low52": round(pct_over_low, 1),
        "pct_under_high52": round(pct_under_high, 1),
        "adr": round(adr, 2),
        "dollar_vol": round(dollar_vol),
        "burst40d": round(burst, 1),
        "rs_raw": rs_raw,
        "setup_a": setup_a,
        "setup_b": setup_b,
        "pivot": pivot,
        "spark": [round(float(x), 2) for x in c.iloc[-40:]],
    }


# ----------------------------------------------------------------------------
# 4. EARNINGS (mejor esfuerzo)
# ----------------------------------------------------------------------------

def days_to_earnings(ticker: str):
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            nxt = min(d for d in dates if d is not None)
            delta = (pd.Timestamp(nxt).tz_localize(None) - pd.Timestamp.now()).days
            if -1 <= delta <= 365:
                return int(delta)
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# 5. SÍMBOLO TRADINGVIEW
# ----------------------------------------------------------------------------

def tv_symbol(ticker: str) -> str:
    if ticker.endswith(".MC"):
        return "BME:" + ticker[:-3]
    if ticker.endswith(".DE"):
        return "XETR:" + ticker[:-3]
    return ticker.replace("-", ".")  # BRK-B -> BRK.B; TradingView resuelve el exchange US


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    universe = get_universe()
    if not universe:
        sys.exit("Universo vacío — revisa la conexión")

    hist = download_history(universe.keys())

    rows = []
    for tk, df in hist.items():
        m = compute_metrics(df)
        if m is None:
            continue
        m["ticker"] = tk
        m["market"] = universe[tk]
        m["tv"] = tv_symbol(tk)
        rows.append(m)

    # Percentil RS sobre todo el universo descargado
    if not rows:
        sys.exit("Descarga vacía: ningún ticker con historial suficiente. "
                 "Revisa la conexión o actualiza yfinance (pip install -U yfinance)")
    raws = sorted(r["rs_raw"] for r in rows)
    n = len(raws)
    for r in rows:
        rank = sum(1 for x in raws if x <= r["rs_raw"])
        r["rs"] = round(rank / n * 100, 1)
        del r["rs_raw"]

    # Earnings solo para los que pasan el trend template (ahorra llamadas)
    candidates = [r for r in rows if r["trend_template"]]
    print(f"Consultando earnings de {len(candidates)} candidatos…")
    for i, r in enumerate(candidates, 1):
        r["days_to_earnings"] = days_to_earnings(r["ticker"])
        if i % 25 == 0:
            print(f"  {i}/{len(candidates)}")
    for r in rows:
        r.setdefault("days_to_earnings", None)

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": len(rows),
        "markets": sorted({r["market"] for r in rows}),
        "stocks": sorted(rows, key=lambda r: -r["rs"]),
    }
    with open("swing_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    mins = (time.time() - t0) / 60
    print(f"\nOK -> swing_data.json  ({len(rows)} valores, {mins:.1f} min)")
    print("Carga el fichero en swing_scanner.html")


if __name__ == "__main__":
    main()
