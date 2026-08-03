# -*- coding: utf-8 -*-
"""
backtest_medias.py — Test de la hipótesis "medias alineadas = el precio sube"
=============================================================================
Regla testeada:
  ENTRADA: cierre cruza a situación de alineación completa
           precio > MA10 > MA15 > MA20 > MA30 (todas SMA diarias)
  SALIDA:  primer cierre por debajo de la MA20

Sobre el mismo universo automático del fetcher (S&P 500, Nasdaq-100,
IBEX 35, DAX 40), 5 años de historia diaria.

Qué responde:
  1. ¿Gana dinero la regla? (tasa de acierto, expectancia, profit factor)
  2. ¿Bate a comprar y mantener el mismo valor?
  3. ¿Funciona mejor en megacaps tranquilas (ADR bajo) o en valores
     con ADR alto? -> desglose por terciles de ADR

Uso:
    pip install yfinance pandas lxml
    python backtest_medias.py

Salida: resumen en consola + backtest_medias.csv (todas las operaciones)
"""

import sys
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Falta yfinance. Instala con: pip install yfinance pandas lxml")

# Reutilizamos el universo automático del fetcher
from fetcher_swing import get_universe, download_history

YEARS = 5


def backtest_ticker(tk: str, df: pd.DataFrame):
    """Devuelve lista de operaciones [(fecha_in, fecha_out, ret_pct, dias)]."""
    c = df["Close"]
    ma10, ma15 = c.rolling(10).mean(), c.rolling(15).mean()
    ma20, ma30 = c.rolling(20).mean(), c.rolling(30).mean()
    aligned = (c > ma10) & (ma10 > ma15) & (ma15 > ma20) & (ma20 > ma30)
    exit_sig = c < ma20

    trades = []
    in_pos = False
    entry_px = entry_i = None
    idx = df.index
    # entramos/salimos al CIERRE de la señal (optimista: sin deslizamiento)
    for i in range(31, len(c)):
        if not in_pos and aligned.iloc[i] and not aligned.iloc[i - 1]:
            in_pos, entry_px, entry_i = True, float(c.iloc[i]), i
        elif in_pos and exit_sig.iloc[i]:
            ret = (float(c.iloc[i]) / entry_px - 1) * 100
            trades.append((tk, idx[entry_i].date(), idx[i].date(), ret, i - entry_i))
            in_pos = False
    # posición abierta al final: se cierra al último precio
    if in_pos:
        ret = (float(c.iloc[-1]) / entry_px - 1) * 100
        trades.append((tk, idx[entry_i].date(), idx[-1].date(), ret, len(c) - 1 - entry_i))
    return trades


def adr20(df: pd.DataFrame) -> float:
    return float(((df["High"] / df["Low"] - 1).iloc[-250:].mean()) * 100)


def buy_hold(df: pd.DataFrame) -> float:
    c = df["Close"]
    return (float(c.iloc[-1]) / float(c.iloc[0]) - 1) * 100


def strategy_return(trades) -> float:
    """Retorno compuesto encadenando las operaciones del ticker."""
    eq = 1.0
    for t in trades:
        eq *= 1 + t[3] / 100
    return (eq - 1) * 100


def stats(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {}
    wins = rows[rows.ret > 0]
    losses = rows[rows.ret <= 0]
    gross_w = wins.ret.sum()
    gross_l = abs(losses.ret.sum())
    return {
        "operaciones": len(rows),
        "acierto_%": round(len(wins) / len(rows) * 100, 1),
        "ganancia_media_%": round(wins.ret.mean(), 2) if len(wins) else 0,
        "perdida_media_%": round(losses.ret.mean(), 2) if len(losses) else 0,
        "expectancia_%": round(rows.ret.mean(), 2),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l else float("inf"),
        "dias_medios": round(rows.dias.mean(), 1),
    }


def main():
    universe = get_universe()
    hist = download_history(universe.keys(), period="10y")

    all_trades, per_ticker = [], []
    for tk, df in hist.items():
        tr = backtest_ticker(tk, df)
        if not tr:
            continue
        all_trades.extend(tr)
        per_ticker.append({
            "ticker": tk, "market": universe[tk], "adr": adr20(df),
            "n_trades": len(tr),
            "ret_estrategia_%": round(strategy_return(tr), 1),
            "ret_buyhold_%": round(buy_hold(df), 1),
        })

    trades = pd.DataFrame(all_trades, columns=["ticker", "entrada", "salida", "ret", "dias"])
    tickers = pd.DataFrame(per_ticker)
    trades.to_csv("backtest_medias.csv", index=False)

    print("\n" + "=" * 64)
    print(f"REGLA: precio>MA10>MA15>MA20>MA30 al cierre / salida cierre<MA20")
    print(f"Universo: {len(tickers)} valores · {YEARS} años · {len(trades)} operaciones")
    print("=" * 64)

    print("\n--- GLOBAL ---")
    for k, v in stats(trades).items():
        print(f"  {k:>18}: {v}")

    # Desglose por terciles de ADR
    tickers["tercil_adr"] = pd.qcut(tickers.adr, 3, labels=["ADR bajo", "ADR medio", "ADR alto"])
    adr_map = dict(zip(tickers.ticker, tickers.tercil_adr))
    trades["tercil_adr"] = trades.ticker.map(adr_map)

    print("\n--- POR VOLATILIDAD (terciles de ADR) ---")
    for lbl in ["ADR bajo", "ADR medio", "ADR alto"]:
        sub = trades[trades.tercil_adr == lbl]
        rango = tickers[tickers.tercil_adr == lbl].adr
        print(f"\n  {lbl}  (ADR {rango.min():.1f}%–{rango.max():.1f}%)")
        for k, v in stats(sub).items():
            print(f"    {k:>18}: {v}")

    print("\n--- ESTRATEGIA vs COMPRAR Y MANTENER (mediana por valor) ---")
    med_e = tickers["ret_estrategia_%"].median()
    med_b = tickers["ret_buyhold_%"].median()
    beat = (tickers["ret_estrategia_%"] > tickers["ret_buyhold_%"]).mean() * 100
    print(f"  Estrategia (mediana): {med_e:.1f}%")
    print(f"  Buy & Hold (mediana): {med_b:.1f}%")
    print(f"  La regla bate a B&H en el {beat:.0f}% de los valores")

    print("\n--- TUS EJEMPLOS ---")
    for tk in ["MSFT", "AMZN", "CAT"]:
        row = tickers[tickers.ticker == tk]
        if not row.empty:
            r = row.iloc[0]
            sub = stats(trades[trades.ticker == tk])
            print(f"  {tk}: {r.n_trades} ops · acierto {sub.get('acierto_%','-')}% · "
                  f"estrategia {r['ret_estrategia_%']}% vs B&H {r['ret_buyhold_%']}%")

    print("\nDetalle completo de operaciones -> backtest_medias.csv")
    print("Nota: sin comisiones ni deslizamiento; entradas/salidas al cierre")
    print("de la señal. Los resultados reales serían algo peores.\n")


if __name__ == "__main__":
    main()
