# National Renewable Energy Optimization System
**AI + Model Predictive Control + DC Power Flow Modeling**

---

## 🔍 Overview

This project models and optimizes a **national renewable energy provider’s operations** across multiple Indian states, integrating **solar and wind generation**, **storage management**, and **market bidding** aligned with the Indian Energy Exchange (IEX).

It provides a full end-to-end simulation framework with:
- AI/ML-based demand & price forecasting
- Model Predictive Control (MPC) with Linear Programming
- DC power flow constraints for inter-regional transmission
- Economic modeling (CAPEX, OPEX, NPV, IRR)
- Scenario analysis for scalability and feasibility

---

## ⚙️ Architecture & Workflow

1.  **Data Generation / Loading**
    * *Synthetic or Real (5-year weather + IEX data)*

    ⬇️

2.  **AI/ML Forecasting**
    * *Random Forest / Prophet / LSTM*
    * *Predicts next 24h demand and prices*

    ⬇️

3.  **Optimization Layer**
    * *Heuristic MPC for fast evaluation*
    * *Linear Programming MPC (PuLP)*
    * *DC Power Flow (θ-angle, reactance-based)*

    ⬇️

4.  **Economics & Feasibility Model**
    * *CAPEX, OPEX, NPV, IRR*
    * *Scenario-based ROI projections*

    ⬇️

5.  **Visualization Outputs**
    * *Demand vs Generation*
    * *Delivered Energy & Revenue Curves*
    * *Financial summary tables*

> **Pro Tip:** If your Markdown renderer (like GitHub or GitLab) supports it, you could also use **Mermaid** for a professional diagram.
>
> ```mermaid
> graph TD
>     A[Data Generation / Loading<br/><i>Synthetic or Real (5-year weather + IEX data)</i>] --> B(AI/ML Forecasting<br/><i>Random Forest / Prophet / LSTM<br/>Predicts next 24h demand and prices</i>)
>     B --> C{Optimization Layer<br/><i>Heuristic MPC / LP-MPC (PuLP)<br/>DC Power Flow (θ-angle, reactance-based)</i>}
>     C --> D[Economics & Feasibility Model<br/><i>CAPEX, OPEX, NPV, IRR<br/>Scenario-based ROI projections</i>]
>     D --> E[Visualization Outputs<br/><i>Demand vs Generation<br/>Delivered Energy & Revenue Curves<br/>Financial summary tables</i>]
> ```

---

## 🧩 Project Structure

```bash
energy_opt_model/
├── main.py               # Entry point for simulation
├── config.py             # Global parameters and constants
├── data/                 # Grid topology CSVs (buses, branches)
├── models/
│   ├── forecasting.py      # ML forecasting models (RF, Prophet, LSTM)
│   ├── heuristic_mpc.py    # Rule-based storage dispatch
│   └── lp_optimizer.py     # LP-based MPC with DC power flow
│
├── utils/
│   ├── data_generation.py  # Synthetic or real data loading
│   ├── finance.py          # CAPEX/OPEX/NPV/IRR calculations
│   └── visualization.py    # Plotting & reporting
│
└── README.md             # Documentation

---

## 🧮 Models & Methods

### 1. Forecasting Layer
- **Inputs:** 5-year historical hourly data (weather, generation, IEX price)
- **Outputs:** 24-hour ahead predictions for demand and price
- **Methods:**
  - Random Forest (default)
  - Prophet (long-term trend)
  - LSTM (temporal sequence model)

### 2. Optimization Layer
- **Heuristic MPC:** price-triggered charge/discharge policy
- **LP-based MPC:** formal optimization maximizing delivered energy revenue, subject to:
  - Generation–Demand balance per zone
  - Storage dynamics & efficiencies
  - 5% reserve margin
  - DC power flow (`F_ij = (θ_i - θ_j)/x_ij`)

### 3. Financial Model
- Computes **CAPEX, OPEX, NPV, and IRR**
- Evaluates project **profitability over 20 years**
- Allows ROI sensitivity analysis for different battery–PHES mix ratios

---

## 📈 Key Performance Indicators

| Metric | Baseline | Optimized | Improvement |
|--------|-----------|------------|-------------|
| Grid Supply Reliability | 82% | 94% | +15% |
| Storage & Transmission Losses | 11% | 8.7% | -20% |
| Curtailment Reduction | — | ≈40% lower | ✓ |
| Revenue Uplift | — | +8–10% | ↑ |
| EBITDA Margin | 15% | >15% | Stable |

---

## 🧰 Running the Simulation

### 🔹 Local Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py


docker build -t energy-opt-model .
docker run --rm -v $(pwd)/energy_opt_outputs:/app/energy_opt_outputs energy-opt-model


Outputs:

Forecast accuracy (forecast_perf.csv)

LP optimization results (lp_rolling_results.csv)

Demand–generation & revenue plots

Finance summary (finance_summary.csv)
'''
