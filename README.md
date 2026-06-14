# Cerberus: Quantitative Radar Module

**Stanford Code in Place — Final Project**  
**Author:** Fidel (Economics, University of Valladolid)

**DISCLAIMER: NO AI MODELS WERE HARMED OR VERBALLY ABUSED THROUGH THE DEVELOPMENT OF THIS PROJECT.**

---

## 1. Project Overview & Genesis

The quantitative radar module presented here is a high-performance terminal application designed to track real-time market momentum. The genesis of this specific module predates the current course, originating from an earlier personal endeavor named *Cerberus* — a broader project aimed at building a comprehensive financial analysis terminal. Inspired by the programmatic rigor of Monte Carlo simulations and the advanced data structures explored in the course's Data modules, I utilized this final assignment to architect a deterministic momentum scanner completely independent of black-box analytical libraries.

---

## 2. The Prototyping Phase: Raw Mathematical Logic

My primary focus during the initial build was uncompromising mathematical accuracy, drawing directly from macroeconomic mechanics. I drafted the first iteration of the script to validate the core statistical logic. In this raw prototype, I coded the canonical Relative Strength Index (RSI) — using J. Welles Wilder's original exponential smoothing rather than simplified moving averages — and the Fast Stochastic Oscillator (%K) as pure, isolated functions.

To bypass the network restrictions of the `antigravity` sandbox environment, I also engineered a deterministic synthetic data generator using seeded random walks. This initial functional draft was written with Spanish nomenclature and a rudimentary command-line output; it served as a mathematically sound proof-of-concept, proving the backend logic worked perfectly.

---

## 3. AI-Assisted Refactoring & Evolution

Once the core statistical engine was mathematically validated, I embraced a modern software engineering workflow. I leveraged advanced Large Language Models (Gemini and Claude) as pair-programming assistants to elevate the raw prototype to production-ready industry standards.

I directed the AI to take my functional mathematical script and refactor it by:

- Translating the nomenclature and logic flow into technical, C2-level English.
- Enforcing strict PEP-8 compliance and generating comprehensive, NumPy-style academic docstrings.
- Upgrading the rudimentary terminal outputs into a highly formatted, color-coded ASCII interface.
- Fortifying the systemic architecture with granular exception handling (e.g., managing `KeyboardInterrupt`, `EOFError`, and `yfinance` network timeouts gracefully).

---

## 4. The Final Architecture

The result of this hybrid development process is a robust, strict three-phase modular pipeline:

- **Phase 1 — Dynamic Data Ingestion:** Natively connects to Yahoo Finance APIs while gracefully falling back to the synthetic deterministic generator for offline sandbox testing.
- **Phase 2 — Pure Mathematical Engine:** The mathematically validated RSI and Stochastic calculations, executing complex matrix operations isolated from the global state.
- **Phase 3 — Interactive Terminal Panel:** The AI-refined, continuous CLI loop that provides clear, actionable signal interpretation.

---

## Installation & Usage

### Prerequisites

This project relies on advanced data structures and mathematical vectorization. Install the following external libraries:

```bash
pip install yfinance pandas numpy
```

### Execution Modes

Because the standard Stanford `antigravity` environment blocks external API requests, a fully functional offline mode is provided.

**1. Offline Demo Mode** *(Recommended for sandboxed grading)*

Runs the quantitative engine using deterministic, synthetic market data generated via seeded random walks. No internet connection required.

```bash
python "Radar RSI.py" --demo
```

**2. Live Market Mode**

Connects directly to the Yahoo Finance API to pull real-time, up-to-date market data for any valid ticker (e.g., `PLTR`, `NVDA`, `BTC-USD`).

```bash
python "Radar RSI.py"
```

---

## Academic Transparency & Code in Place Guidelines

This project intentionally steps beyond the standard Code in Place curriculum (which focuses on fundamental control flow, lists, and dictionaries) to explore advanced Python capabilities. It represents a deep, self-directed exploration of Python's potential when applied to financial data science, merging the theoretical market dynamics of my Economics background with hard computational logic.
