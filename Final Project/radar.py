"""
radar.py — CERBERUS Market Scanner
Indicadores técnicos: Estocástico %K (14) + RSI (14)
Fuente de datos: Yahoo Finance via yfinance
Uso: python radar.py
     python radar.py --demo   <- modo offline sin conexión
"""

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime


# colores ANSI para la terminal
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# configuración de los indicadores
PERIOD    = 14
HISTORY   = "3mo"
INTERVAL  = "1d"

# umbrales de señal
STOCH_OVERSOLD  = 20.0
STOCH_OVERBOUGHT= 80.0
RSI_OVERSOLD    = 30.0
RSI_OVERBOUGHT  = 70.0


# =============================================================================
# FASE 1 — INGESTA DE DATOS
# =============================================================================

def fetch_ohlc(ticker: str) -> pd.DataFrame:
    """Descarga velas diarias de Yahoo Finance y devuelve High, Low, Close."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Instala yfinance: pip install yfinance")

    obj = yf.Ticker(ticker)
    raw = obj.history(period=HISTORY, interval=INTERVAL)

    if raw.empty:
        raise ValueError(f"Sin datos para '{ticker}'. Comprueba el símbolo.")

    ohlc = raw[["High", "Low", "Close"]].dropna()

    if len(ohlc) < PERIOD + 1:
        raise ValueError(
            f"Datos insuficientes para '{ticker}': {len(ohlc)} sesiones."
        )

    return ohlc


def demo_ohlc(ticker: str, n: int = 60) -> pd.DataFrame:
    """Datos OHLC sintéticos para testear sin red. Seed basado en el ticker."""
    seed = sum(ord(c) for c in ticker.upper())
    rng  = np.random.default_rng(seed)

    # random walk alrededor de 100 — parámetros típicos de equity
    returns = rng.normal(0.0003, 0.018, n)
    closes  = 100.0 * np.cumprod(1 + returns)
    highs   = closes + rng.uniform(0.5, 2.5, n)
    lows    = closes - rng.uniform(0.5, 2.5, n)

    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=idx)


# =============================================================================
# FASE 2 — MOTOR MATEMÁTICO (funciones puras)
# =============================================================================

def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """
    Calcula el Oscilador Estocástico %K de 14 periodos.

    Fórmula: %K = (C - Min_L14) / (Max_H14 - Min_L14) * 100
    """
    max_high   = high.iloc[-PERIOD:].max()
    min_low    = low.iloc[-PERIOD:].min()
    last_close = float(close.iloc[-1])

    price_range = max_high - min_low
    if price_range == 0:
        return 50.0  # mercado plano, denominador cero

    k = (last_close - min_low) / price_range * 100.0
    return round(k, 2)


def compute_rsi(close: pd.Series) -> float:
    """
    Calcula el RSI de 14 periodos con el suavizado exponencial de Wilder.

    Fase 1 (semilla): media simple de las primeras 14 variaciones diarias.
    Fase 2 (smoothing): avg = (avg_prev * 13 + valor_actual) / 14
    Esto es el RSI canónico de Wilder, no la aproximación de SMA que
    está mal implementada en la mayoría de ejemplos de internet.
    """
    delta  = close.diff().dropna()
    gains  = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # semilla con las primeras 14 velas
    avg_gain = float(gains.iloc[:PERIOD].mean())
    avg_loss = float(losses.iloc[:PERIOD].mean())

    # suavizado de Wilder para el resto de sesiones
    for i in range(PERIOD, len(gains)):
        avg_gain = (avg_gain * (PERIOD - 1) + float(gains.iloc[i])) / PERIOD
        avg_loss = (avg_loss * (PERIOD - 1) + float(losses.iloc[i])) / PERIOD

    if avg_loss == 0:
        return 100.0  # tendencia alcista pura, sin pérdidas

    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


# =============================================================================
# FASE 3 — PANEL DE CONTROL
# =============================================================================

def label_stoch(k: float) -> str:
    if k <= STOCH_OVERSOLD:
        return f"{GREEN}ALERTA: ZONA DE SOBREVENTA{RESET}"
    if k >= STOCH_OVERBOUGHT:
        return f"{RED}ALERTA: ZONA DE SOBRECOMPRA{RESET}"
    return f"{YELLOW}NEUTRAL{RESET}"


def label_rsi(r: float) -> str:
    if r <= RSI_OVERSOLD:
        return f"{GREEN}ALERTA: ZONA DE SOBREVENTA{RESET}"
    if r >= RSI_OVERBOUGHT:
        return f"{RED}ALERTA: ZONA DE SOBRECOMPRA{RESET}"
    return f"{YELLOW}NEUTRAL{RESET}"


def print_report(ticker: str, k: float, r: float, price: float, demo: bool = False):
    sep      = "=" * 52
    demo_tag = f"  {YELLOW}[DEMO]{RESET}" if demo else ""
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")

    print()
    print(sep)
    print(f"  {BOLD}{CYAN}[ {ticker} ]{RESET}{demo_tag}  |  ${price:.2f}  |  {ts}")
    print(sep)
    print(f"  -> ESTOCÁSTICO (14): {BOLD}{k:>6.1f}{RESET}   [{label_stoch(k)}]")
    print(f"  -> RSI         (14): {BOLD}{r:>6.1f}{RESET}   [{label_rsi(r)}]")
    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(description="CERBERUS — Radar de indicadores técnicos")
    parser.add_argument("--demo", action="store_true", help="Modo offline con datos sintéticos")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}  CERBERUS — RADAR  |  v1.0{RESET}")
    print(f"  Indicadores: Estocástico (14)  ·  RSI (14)")
    if args.demo:
        print(f"  {YELLOW}[MODO DEMO — datos sintéticos]{RESET}")
    print(f"  Escribe 'exit' para salir.\n")

    while True:
        try:
            raw_input = input("  Introduce el ticker a escanear: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {CYAN}[CERBERUS]{RESET} Sesión cerrada.\n")
            sys.exit(0)

        if not raw_input:
            print("  [!] Introduce un símbolo válido (ej: PLTR, NVDA, BTC-USD)\n")
            continue

        if raw_input in ("EXIT", "QUIT", "Q"):
            print(f"\n  {CYAN}[CERBERUS]{RESET} Sesión cerrada.\n")
            sys.exit(0)

        print(f"  Descargando datos de {BOLD}{raw_input}{RESET}...")

        try:
            ohlc = demo_ohlc(raw_input) if args.demo else fetch_ohlc(raw_input)

            k_val = stochastic_k(ohlc["High"], ohlc["Low"], ohlc["Close"])
            r_val = compute_rsi(ohlc["Close"])
            price = float(ohlc["Close"].iloc[-1])

            print_report(raw_input, k_val, r_val, price, demo=args.demo)

        except (ValueError, ImportError) as e:
            print(f"\n  [ERROR] {e}\n")
        except Exception as e:
            print(f"\n  [ERROR] {e}\n")


if __name__ == "__main__":
    main()
