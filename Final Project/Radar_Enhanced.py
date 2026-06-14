"""
╔══════════════════════════════════════════════════════════════════╗
║              CERBERUS — QUANTITATIVE RADAR MODULE                ║
║                     radar.py  |  v1.0                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Architecture  : Three-phase pipeline                            ║
║    Phase 1     : Real market data ingestion  (yfinance)          ║
║    Phase 2     : Pure mathematical engine    (Stochastic, RSI)   ║
║    Phase 3     : Interactive terminal control panel              ║
╠══════════════════════════════════════════════════════════════════╣
║  Dependencies  : pip install yfinance pandas numpy               ║
║  Usage         : python radar.py                                 ║
║                  python radar.py --demo    (offline test mode)   ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

STOCH_PERIOD    : int = 14       # Look-back window for the Stochastic Oscillator
RSI_PERIOD      : int = 14       # Look-back window for the RSI

DATA_PERIOD     : str = "3mo"    # Historical span — 3 months is the minimum
DATA_INTERVAL   : str = "1d"     # Daily candlestick resolution

# Signal threshold boundaries
OVERSOLD_STOCH  : float = 20.0
OVERBOUGHT_STOCH: float = 80.0
OVERSOLD_RSI    : float = 30.0
OVERBOUGHT_RSI  : float = 70.0

# Terminal colour codes (gracefully degraded to plain if unsupported)
try:
    import os
    _USE_COLOUR = sys.stdout.isatty() or os.getenv("FORCE_COLOR")
except Exception:
    _USE_COLOUR = False

_RED    = "\033[91m"  if _USE_COLOUR else ""
_GREEN  = "\033[92m"  if _USE_COLOUR else ""
_YELLOW = "\033[93m"  if _USE_COLOUR else ""
_CYAN   = "\033[96m"  if _USE_COLOUR else ""
_BOLD   = "\033[1m"   if _USE_COLOUR else ""
_RESET  = "\033[0m"   if _USE_COLOUR else ""


# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — DATA INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def fetch_ohlc(ticker: str) -> pd.DataFrame:
    """
    Download daily OHLC candlestick data from Yahoo Finance via yfinance.

    The function uses ``yf.Ticker.history()`` (preferred over ``yf.download``)
    because it always returns a flat, single-level column index regardless of
    the installed yfinance version, making it robust against API changes.

    Parameters
    ----------
    ticker : str
        Market symbol accepted by Yahoo Finance
        (e.g. "PLTR", "AAPL", "NVDA", "BTC-USD", "GC=F" for Gold futures).

    Returns
    -------
    pd.DataFrame
        Indexed by date.  Guaranteed columns: ``High``, ``Low``, ``Close``.
        All NaN rows are dropped before returning.

    Raises
    ------
    ImportError
        If yfinance is not installed in the current environment.
    ValueError
        If the ticker is unrecognised or the downloaded series is too short
        to warm up the indicators (minimum: RSI_PERIOD + 1 sessions).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is not installed.  Run:  pip install yfinance"
        ) from exc

    obj = yf.Ticker(ticker)
    raw = obj.history(period=DATA_PERIOD, interval=DATA_INTERVAL)

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for '{ticker}'.  "
            "Verify the symbol is valid and that your network can reach "
            "query2.finance.yahoo.com."
        )

    # Normalise: keep only the three columns we need and drop incomplete rows
    ohlc = raw[["High", "Low", "Close"]].dropna()

    min_rows_required = RSI_PERIOD + 1   # need at least 15 sessions for RSI
    if len(ohlc) < min_rows_required:
        raise ValueError(
            f"Insufficient history for '{ticker}': "
            f"received {len(ohlc)} sessions, need ≥ {min_rows_required}."
        )

    return ohlc


def _build_demo_ohlc(ticker: str, n: int = 60) -> pd.DataFrame:
    """
    Generate a synthetic OHLC DataFrame for offline testing.

    Uses a seeded random walk so results are deterministic and reproducible.
    The seed is derived from the ticker string so each symbol produces a
    unique (but consistent) dataset across runs.

    Parameters
    ----------
    ticker : str
        Used as the random seed source.
    n : int
        Number of synthetic daily sessions to generate.

    Returns
    -------
    pd.DataFrame
        Columns: ``High``, ``Low``, ``Close``.
    """
    seed = sum(ord(c) for c in ticker.upper())
    rng  = np.random.default_rng(seed)

    # Build the date index first.  pandas 3.x with freq="B" may yield fewer
    # entries than `n` when today is a weekend or public holiday, so we derive
    # the actual row count from the index length to keep arrays aligned.
    dates    = pd.date_range(end=pd.Timestamp.today().normalize(),
                             periods=n, freq="B")
    actual_n = len(dates)

    # Random walk for Close prices anchored around 100
    returns = rng.normal(loc=0.0003, scale=0.018, size=actual_n)
    closes  = 100.0 * np.cumprod(1 + returns)
    highs   = closes + rng.uniform(0.5, 2.5, size=actual_n)
    lows    = closes - rng.uniform(0.5, 2.5, size=actual_n)

    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes},
                        index=dates)


# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — MATHEMATICAL ENGINE  (pure functions)
# ──────────────────────────────────────────────────────────────────────────────

def compute_stochastic_k(
    high  : pd.Series,
    low   : pd.Series,
    close : pd.Series,
    period: int = STOCH_PERIOD,
) -> float:
    """
    Compute the raw Fast Stochastic Oscillator (%K) for the last session.

    Formula
    -------
    .. math::

        \\%K = \\frac{C_t - \\min(L_{t-n+1},\\ldots,L_t)}
                     {\\max(H_{t-n+1},\\ldots,H_t) - \\min(L_{t-n+1},\\ldots,L_t)}
              \\times 100

    where:
        - :math:`C_t`   = most recent closing price
        - :math:`L_n`   = lowest Low over the last ``period`` sessions
        - :math:`H_n`   = highest High over the last ``period`` sessions

    Parameters
    ----------
    high   : pd.Series   Daily High prices  (length ≥ ``period``)
    low    : pd.Series   Daily Low prices   (length ≥ ``period``)
    close  : pd.Series   Daily Close prices (length ≥ ``period``)
    period : int         Look-back window   (default: 14)

    Returns
    -------
    float
        %K value in the closed interval [0, 100].
        Returns 50.0 when the price range within the window is zero
        (perfectly flat market — undefined denominator).
    """
    if len(close) < period:
        raise ValueError(
            f"stochastic_k requires at least {period} data points; "
            f"received {len(close)}."
        )

    window_high    = high.iloc[-period:]
    window_low     = low.iloc[-period:]
    current_close  = float(close.iloc[-1])

    highest_high   = float(window_high.max())
    lowest_low     = float(window_low.min())
    price_range    = highest_high - lowest_low

    if price_range == 0.0:
        # Flat market — stochastic is mathematically undefined; return midpoint
        return 50.0

    k = (current_close - lowest_low) / price_range * 100.0
    return round(k, 2)


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    """
    Compute the Relative Strength Index using Wilder's exponential smoothing.

    This is the *canonical* RSI as defined by J. Welles Wilder (1978), not the
    simple-average approximation.  The algorithm has two stages:

    **Stage 1 — Seed** (sessions 1 … period)
        The initial average gain and loss are the arithmetic mean of the first
        ``period`` non-zero absolute price changes.

    **Stage 2 — Smooth** (session period+1 … end)
        Wilder's exponential smoothing with a smoothing factor of ``1/period``:

        .. math::
            \\overline{G}_t = \\frac{\\overline{G}_{t-1} \\times (n-1) + G_t}{n}

        where :math:`G_t = \\max(\\Delta C_t,\\, 0)`.  Losses are computed
        symmetrically.

    **Final**

    .. math::
        RS  = \\frac{\\overline{G}}{\\overline{L}}, \\quad
        RSI = 100 - \\frac{100}{1 + RS}

    Edge cases:
        - :math:`\\overline{L} = 0` → all sessions were up-days → RSI = 100
        - :math:`\\overline{G} = 0` → all sessions were down-days → RSI = 0

    Parameters
    ----------
    close  : pd.Series   Daily Close prices  (length ≥ ``period`` + 1)
    period : int         Look-back window    (default: 14)

    Returns
    -------
    float
        RSI value in the closed interval [0, 100].

    Raises
    ------
    ValueError
        If ``close`` has fewer than ``period + 1`` data points.
    """
    if len(close) < period + 1:
        raise ValueError(
            f"compute_rsi requires ≥ {period + 1} data points; "
            f"received {len(close)}."
        )

    # Daily price deltas — length = len(close) - 1
    delta = close.diff().dropna()

    gains  = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # ── Stage 1: seed with the simple mean of the first `period` bars ────────
    avg_gain: float = float(gains.iloc[:period].mean())
    avg_loss: float = float(losses.iloc[:period].mean())

    # ── Stage 2: Wilder's exponential smoothing for all subsequent bars ───────
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses.iloc[i])) / period

    # ── Edge-case guards ──────────────────────────────────────────────────────
    if avg_loss == 0.0:
        return 100.0   # Pure uptrend — maximum strength
    if avg_gain == 0.0:
        return 0.0     # Pure downtrend — minimum strength

    rs        = avg_gain / avg_loss
    rsi_value = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi_value, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS — SIGNAL INTERPRETATION  (pure functions)
# ──────────────────────────────────────────────────────────────────────────────

def _signal_stoch(k: float) -> tuple[str, str]:
    """Return (label, colour_code) for a given %K value."""
    if k <= OVERSOLD_STOCH:
        return "ALERT: OVERSOLD ZONE ", _GREEN
    if k >= OVERBOUGHT_STOCH:
        return "ALERT: OVERBOUGHT ZONE", _RED
    return "NEUTRAL               ", _YELLOW


def _signal_rsi(r: float) -> tuple[str, str]:
    """Return (label, colour_code) for a given RSI value."""
    if r <= OVERSOLD_RSI:
        return "ALERT: OVERSOLD ZONE ", _GREEN
    if r >= OVERBOUGHT_RSI:
        return "ALERT: OVERBOUGHT ZONE", _RED
    return "NEUTRAL               ", _YELLOW


# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — CONTROL PANEL  (terminal output)
# ──────────────────────────────────────────────────────────────────────────────

_W = 56   # Report panel width


def _hline(char: str = "─") -> str:
    return char * _W


def _box_line(content: str) -> str:
    """Pad content to panel width and wrap in box characters."""
    return f"│ {content:<{_W - 2}} │"


def print_report(
    ticker    : str,
    k_value   : float,
    rsi_value : float,
    last_close: float,
    timestamp : str,
    demo_mode : bool = False,
) -> None:
    """
    Render the formatted indicator report to stdout.

    Parameters
    ----------
    ticker     : str   Ticker symbol (will be uppercased).
    k_value    : float Stochastic %K result.
    rsi_value  : float RSI result.
    last_close : float Most recent closing price.
    timestamp  : str   Formatted scan datetime string.
    demo_mode  : bool  If True, appends a [DEMO] badge to the header.
    """
    k_label,   k_color   = _signal_stoch(k_value)
    rsi_label, rsi_color = _signal_rsi(rsi_value)

    demo_badge = f"  {_YELLOW}[DEMO DATA]{_RESET}" if demo_mode else ""

    print()
    print(f"┌{_hline()}┐")
    print(f"│{_BOLD}{_CYAN}{'  CERBERUS — MARKET SCANNER':^{_W}}{_RESET}│")
    print(f"├{_hline()}┤")
    print(_box_line(
        f"  TICKER    : {_BOLD}{ticker.upper()}{_RESET}{demo_badge}"
    ))
    print(_box_line(f"  PRICE     : ${last_close:.4f}"))
    print(_box_line(f"  SCANNED   : {timestamp}"))
    print(f"├{_hline()}┤")
    print(_box_line(
        f"  -> STOCHASTIC (14) : "
        f"{_BOLD}{k_value:>6.1f}{_RESET}  "
        f"[{k_color}{k_label}{_RESET}]"
    ))
    print(_box_line(
        f"  -> RSI       (14) : "
        f"{_BOLD}{rsi_value:>6.1f}{_RESET}  "
        f"[{rsi_color}{rsi_label}{_RESET}]"
    ))
    print(f"└{_hline()}┘")
    print()


def _print_banner(demo_mode: bool) -> None:
    mode_label = (
        f"  {_YELLOW}MODE: OFFLINE DEMO  (synthetic data){_RESET}"
        if demo_mode
        else f"  {_GREEN}MODE: LIVE          (Yahoo Finance / yfinance){_RESET}"
    )

    print()
    print("═" * (_W + 2))
    print(f"  {_BOLD}{_CYAN}CERBERUS — QUANTITATIVE RADAR  |  v1.0{_RESET}")
    print(f"  Turbo Investments — Internal Tool")
    print("─" * (_W + 2))
    print(f"  Indicators : Stochastic %K ({STOCH_PERIOD})  ·  RSI ({RSI_PERIOD})")
    print(f"  Interval   : Daily candles  |  History: {DATA_PERIOD}")
    print(mode_label)
    print("─" * (_W + 2))
    print(f"  Type {_BOLD}exit{_RESET} or {_BOLD}quit{_RESET} to terminate.")
    print("═" * (_W + 2))
    print()


def _run_scan(ticker: str, demo_mode: bool) -> None:
    """
    Execute the full three-phase pipeline for a single ticker and print the
    report.  All errors are propagated to the caller.

    Parameters
    ----------
    ticker    : str
    demo_mode : bool
    """
    # ── Phase 1: Ingestion ───────────────────────────────────────────────────
    if demo_mode:
        ohlc = _build_demo_ohlc(ticker)
    else:
        ohlc = fetch_ohlc(ticker)

    high  = ohlc["High"]
    low   = ohlc["Low"]
    close = ohlc["Close"]

    # ── Phase 2: Math Engine ─────────────────────────────────────────────────
    k_value   = compute_stochastic_k(high, low, close)
    rsi_value = compute_rsi(close)

    # ── Phase 3: Report ──────────────────────────────────────────────────────
    last_close = float(close.iloc[-1])
    timestamp  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    print_report(ticker, k_value, rsi_value, last_close, timestamp, demo_mode)


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Interactive terminal control panel — infinite scan loop.

    Accepts one ticker per iteration, runs the full pipeline, and prints the
    report.  Continues until the user types ``exit`` / ``quit``, presses
    Ctrl-C, or sends an EOF (Ctrl-D on Unix).

    CLI flags
    ---------
    --demo    Run in offline mode using deterministic synthetic OHLC data.
              Useful for testing the math engine without a network connection.
    """
    parser = argparse.ArgumentParser(
        prog="radar",
        description="CERBERUS Quantitative Radar — market indicator scanner",
        epilog="Example: python radar.py --demo",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic offline data instead of the live Yahoo Finance API.",
    )
    args = parser.parse_args()

    demo_mode: bool = args.demo

    _print_banner(demo_mode)

    while True:
        try:
            raw_input = input("  >>> Ticker to scan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {_CYAN}[CERBERUS]{_RESET} Session closed.  Stand by.\n")
            sys.exit(0)

        ticker = raw_input.upper()

        if not ticker:
            print(f"  {_YELLOW}[!]{_RESET} Enter a valid ticker symbol "
                  f"(e.g. PLTR, NVDA, BTC-USD).\n")
            continue

        if ticker in {"EXIT", "QUIT", "Q", ":Q", ".EXIT"}:
            print(f"\n  {_CYAN}[CERBERUS]{_RESET} Session terminated.  Stand by.\n")
            sys.exit(0)

        print(f"  {_CYAN}[*]{_RESET} Fetching data for "
              f"{_BOLD}{ticker}{_RESET}  …")

        try:
            _run_scan(ticker, demo_mode)

        except ImportError as exc:
            print(f"\n  {_RED}[DEPENDENCY ERROR]{_RESET}  {exc}\n")
        except ValueError as exc:
            print(f"\n  {_RED}[DATA ERROR]{_RESET}  {exc}\n")
        except Exception as exc:
            print(f"\n  {_RED}[ERROR]{_RESET}  Unexpected: {exc}\n")


if __name__ == "__main__":
    main()