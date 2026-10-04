# -*- coding: utf-8 -*-
"""
fetcher_swing.py — Pipeline de datos para el Swing Scanner
"""

import json
import math
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    sys.exit("Falta yfinance. Instala con: pip install yfinance pandas lxml")

# ----------------------------------------------------------------------------
# 1. UNIVERSO AUTOMÁTICO CON FALLBACKS COMPLETOS
# ----------------------------------------------------------------------------

WIKI = {
    "SP500": (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "Symbol",
    ),
    "NDX": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    "IBEX35": ("https://es.wikipedia.org/wiki/IBEX_35", "Ticker"),
    "DAX40": ("https://en.wikipedia.org/wiki/DAX", "Ticker"),
}

FALLBACK_IBEX35 = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC", "BBVA.MC",
    "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC", "ENG.MC", "FDR.MC",
    "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC", "IDR.MC", "ITX.MC", "LOG.MC",
    "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC", "PUIG.MC", "RED.MC", "REP.MC",
    "ROVI.MC", "SAB.MC", "SAN.MC", "SCYR.MC", "SLR.MC", "TEF.MC", "UNI.MC",
]

FALLBACK_NDX = [
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "ASML", "NFLX", "AMD", "AZN", "PEP", "LIN", "TMUS", "ADBE",
    "CSCO", "PDD", "TXN", "QCOM", "AMAT", "CMCSA", "INTU", "AMGN", "ISRG",
    "HON", "BKNG", "VRTX", "ADP", "REGN", "PANW", "MDLZ", "MU", "LRCX",
    "ADI", "MELI", "KLAC", "GILD", "SNPS", "CDNS", "CRWD", "INTC", "ORLY",
    "CSX", "MAR", "CTAS", "PYPL", "ABNB",
]

FALLBACK_DAX40 = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE",
    "BNR.DE", "CBK.DE", "CON.DE", "1COV.DE", "DTG.DE", "DB1.DE", "DBK.DE",
    "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "HNR1.DE",
    "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PPA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SY1.DE", "VOW3.DE",
    "VNA.DE", "ZAL.DE",
]


def _clean_symbol(sym: str, market: str) -> str:
    sym = str(sym).strip().upper()
    if market in ("SP500", "NDX"):
        return sym.replace(".", "-")
    if market == "IBEX35":
        return sym if sym.endswith(".MC") else sym + ".MC"
    if market == "DAX40":
        return sym if sym.endswith(".DE") else sym + ".DE"
    return sym


def get_universe() -> dict:
    universe = {}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }

    for market, (url, col) in WIKI.items():
        symbols = []
        try:
            html = requests.get(url, headers=headers, timeout=10).text
            tables = pd.read_html(html)
            for t in tables:
                cols = [str(c) for c in t.columns]
                match = next(
                    (c for c in cols if col.lower() in c.lower()), None
                )
                if match and len(t) > 20:
                    symbols = [
                        _clean_symbol(s, market) for s in t[match].dropna()
                    ]
                    break
        except Exception as e:
            print(f"[AVISO] Wikipedia falló para {market}: {e}")

        if not symbols:
            if market == "IBEX35":
                symbols = FALLBACK_IBEX35
            elif market == "NDX":
                symbols = FALLBACK_NDX
            elif market == "DAX40":
                symbols = FALLBACK_DAX40
            elif market == "SP500":
                symbols = FALLBACK_NDX

            print(f"[AVISO] Usando lista de respaldo para {market}")

        for s in symbols:
            universe.setdefault(s, market)
        print(f"{market}: {len(symbols)} componentes")
    return universe


def download_history(tickers, period="2y"):
    print(f"Descargando {len(tickers)} tickers…")
    data = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    out = {}
    for t in tickers:
        try:
            df = (
                data[t].dropna(how="all")
                if len(tickers) > 1
                else data.dropna(how="all")
            )
            df = df.dropna(subset=["Close"])
            if len(df) >= 220:
                out[t] = df
        except Exception:
            continue
    print(f"Con historial suficiente: {len(out)}")
    return out


def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def calc_trend_exhaustion(close, volume):
    """
    SISTEMA UNIFICADO GENERAL DE ALERTA DE AGOTAMIENTO Y DISTRIBUCIÓN
    -----------------------------------------------------------------
    Evalúa 4 patrones institucionales sin esperar a caídas graves (-15%)
    ni saltar por pequeñas pausas de consolidación (-1%).
    """
    if close is None or volume is None or len(close) < 60:
        return 0

    c = close
    v = volume
    vol_ma50 = v.rolling(50).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    hi52 = c.rolling(252, min_periods=60).max()

    rets = c.pct_change()
    rel_vol = v / vol_ma50

    # 1. CLÚSTER DE DISTRIBUCIÓN: Días de caída >= 0.8% con Volumen > 1.25x MA50 en 15 sesiones
    dist_days_15 = int(((rets <= -0.008) & (rel_vol > 1.25)).iloc[-15:].sum())

    # 2. CHURNING / ESTANCAMIENTO: Volumen > 1.5x pero precio plano (-0.5% a +0.3%) cerca de máximos (>= 92% de 52w)
    near_highs = (c / hi52) >= 0.92
    stalling_days = int(((rets >= -0.005) & (rets <= 0.003) & (rel_vol > 1.5) & near_highs).iloc[-10:].sum())

    # 3. AGOTAMIENTO CLIMÁTICO: Precio extendido > 18% sobre EMA21 con volumen alto (> 1.8x)
    ext_ema21 = (c / ema21) - 1.0
    climax_exhaustion = 1 if (ext_ema21.iloc[-1] > 0.18 and rel_vol.iloc[-1] > 1.8) else 0

    # 4. PÉRDIDA DE CARÁCTER: Cierre por debajo de la EMA21 con volumen institucional (> 1.3x)
    character_loss = 1 if (c.iloc[-1] < ema21.iloc[-1] and rel_vol.iloc[-1] > 1.3) else 0

    # --- PONDERACIÓN DEL SCORE DE SALIDA ---
    score = 0
    if dist_days_15 >= 3:
        score += 2
    elif dist_days_15 == 2:
        score += 1

    if stalling_days >= 2:
        score += 1

    if climax_exhaustion:
        score += 1

    if character_loss:
        score += 1

    # ANULACIÓN POR ABSORCIÓN: Si la última sesión es un fuerte rebote alcista (> +2.0%), la distribución se neutraliza
    if rets.iloc[-1] > 0.020:
        score = max(0, score - 2)

    return min(4, score)


def compute_metrics(df: pd.DataFrame) -> dict | None:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    if len(c) < 220:
        return None
    last = float(c.iloc[-1])

    sma50, sma150, sma200 = sma(c, 50), sma(c, 150), sma(c, 200)
    ema10, ema21 = ema(c, 10), ema(c, 21)

    sma200_up = bool(sma200.iloc[-1] > sma200.iloc[-22])

    lo52 = float(l.iloc[-252:].min())
    hi52 = float(h.iloc[-252:].max())
    pct_over_low = (last / lo52 - 1) * 100
    pct_under_high = (last / hi52 - 1) * 100

    trend_template = bool(
        last > sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]
        and sma200_up
        and pct_over_low >= 30
        and pct_under_high >= -25
    )

    adr = float(((h / l - 1).iloc[-20:].mean()) * 100)
    dollar_vol = float((c * v).iloc[-20:].mean())
    roll_min = c.rolling(40).min()
    burst = float(((c / roll_min - 1).iloc[-252:].max()) * 100)

    def ret_func(n):
        return float(c.iloc[-1] / c.iloc[-n] - 1) if len(c) > n else 0.0

    rs_raw = 0.4 * ret_func(63) + 0.2 * ret_func(126) + 0.2 * ret_func(189) + 0.2 * ret_func(252)

    hi13w = float(h.iloc[-65:].max())
    rng_last10 = float(h.iloc[-10:].max() / l.iloc[-10:].min() - 1)
    rng_prev10 = float(h.iloc[-20:-10].max() / l.iloc[-20:-10].min() - 1)
    vol_drying = bool(v.iloc[-10:].mean() < v.iloc[-30:-10].mean())
    setup_a = bool(
        trend_template
        and last >= hi13w * 0.85
        and rng_prev10 > 0
        and rng_last10 < rng_prev10
        and vol_drying
    )
    pivot = round(hi13w, 2)

    made_20d_high_recently = bool(
        (h.iloc[-15:] >= h.rolling(20).max().iloc[-15:]).any()
    )
    touched_ema = bool(
        (l.iloc[-3:] <= ema10.iloc[-3:] * 1.01).any()
        or (l.iloc[-3:] <= ema21.iloc[-3:] * 1.01).any()
    )
    above_ema21 = bool((c.iloc[-10:] > ema21.iloc[-10:] * 0.99).all())
    setup_b = bool(
        trend_template
        and made_20d_high_recently
        and touched_ema
        and above_ema21
        and vol_drying
    )

    # -------------------------------------------------------------
    # CÁLCULO DE DÍAS SECOS UNIFICADO (Volumen < 0.55 y Rango < 0.012)
    # -------------------------------------------------------------
    dry_days = 0
    if len(v) >= 50 and len(c) >= 11:
        v_last10 = v.iloc[-10:]
        v_ma50 = v.rolling(50).mean().iloc[-10:]
        rel_vols = v_last10 / v_ma50
        ret_last10 = c.iloc[-11:].pct_change().dropna().abs()
        
        for i in range(len(rel_vols)):
            if rel_vols.iloc[i] < 0.55 and ret_last10.iloc[i] < 0.012:
                dry_days += 1

    # -------------------------------------------------------------
    # MÉTRICA UNIFICADA DE DISTRIBUCIÓN / AGOTAMIENTO INSTITUCIONAL
    # -------------------------------------------------------------
    heavy_days_count = calc_trend_exhaustion(c, v)

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
        "dryDays10": int(dry_days),
        "heavyDays10": int(heavy_days_count),
        "rs_raw": rs_raw,
        "setup_a": setup_a,
        "setup_b": setup_b,
        "pivot": pivot,
        "spark": [round(float(x), 2) for x in c.iloc[-40:]],
    }


def days_to_earnings(ticker: str):
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            nxt = min(d for d in dates if d is not None)
            delta = (
                pd.Timestamp(nxt).tz_localize(None) - pd.Timestamp.now()
            ).days
            if -1 <= delta <= 365:
                return int(delta)
    except Exception:
        pass
    return None


def tv_symbol(ticker: str) -> str:
    if ticker.endswith(".MC"):
        return "BME:" + ticker[:-3]
    if ticker.endswith(".DE"):
        return "XETR:" + ticker[:-3]
    return ticker.replace("-", ".")


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

    if not rows:
        sys.exit("Descarga vacía")

    raws = sorted(r["rs_raw"] for r in rows)
    n = len(raws)
    for r in rows:
        rank = sum(1 for x in raws if x <= r["rs_raw"])
        r["rs"] = round(rank / n * 100, 1)
        del r["rs_raw"]

    candidates = [r for r in rows if r["trend_template"]]
    print(f"Consultando earnings de {len(candidates)} candidatos…")
    for r in candidates:
        r["days_to_earnings"] = days_to_earnings(r["ticker"])
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
    print(
        f"\nOK -> swing_data.json  ({len(rows)} valores procesados, {mins:.1f} min)"
    )


if __name__ == "__main__":
    main()
