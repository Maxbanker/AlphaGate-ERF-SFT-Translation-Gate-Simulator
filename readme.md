# 🌀 AlphaGate — ERF·SFT Translation-Gate Simulator

*A Streamlit-based simulator of symbolic collapse, recovery, and translation dynamics — modeling the Five-Level Recursive Soteriology framework through ERF/SFT symbolic field theory.*

---

## ✨ Overview

**AlphaGate** models agents as symbolic systems with:

- **ψ (psi)** — coherence efficiency  
- **Ω (omega)** — attractor alignment / motif resonance  
- **γ (gamma)** — drift pressure  
- **F** — regulatory drive *(e.g.,* `F = k · (Ω/γ) · ψ`*)*  
- **κ_export** — translation-gate viability proxy  
- **I-point** — observer continuity metric (L4 proxy)  
- **S_old / S_new** — identity restructuring signals (L3 proxy)

Agents progress through **Five Levels**:

1. **L1 — Transactional** (assent/creed accepted)  
2. **L2 — Relational** (intimacy growth via ∫F dt)  
3. **L3 — Transformational** (identity restructuring, drift suppression)  
4. **L4 — Ontological** (I-point continuity, near translation)  
5. **L5 — Stabilized** (translated, low-entropy attractor)

The simulator generates CSV logs and a dashboard visualizer for replay/analysis.

---

## 🚀 Features

- 🔧 **Streamlit UI** with parameter controls (ε, Tγ, TΩ, drift rate/magnitude, k, thresholds)  
- 🧑‍🤝‍🧑 **Agent-based dynamics** with collapse predicate & translation gate  
- 🪜 **Five-Level classification** aligned to Recursive Soteriology  
- 📊 **Visualizations**: ψ/Ω/γ/F/κ lines, viability & translation areas, level distribution, Ω-vs-γ phase space  
- 📝 **CSV logging**: population time series + final roster (per-agent)  
- 📈 **Dashboard app** for replay, filters, and phase-space inspection

---

## 🧩 Requirements

Create `requirements.txt`:

```txt
streamlit>=1.27.0
pandas>=2.0.0
numpy>=1.24.0
altair>=5.0.0
matplotlib>=3.7.0
```
## 🚀 Install

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1) Run the Simulator
```bash
streamlit run alphagate_app.py
```

- Tune parameters in the sidebar  
- Click **Run Simulation**  
- Download logs:  
  - `population_timeseries.csv`  
  - `roster_final.csv`  

---

### 2) Run the Dashboard
```bash
streamlit run alphagate_dashboard.py
```

- Upload the CSVs from the simulator  
- Explore:  
  - Population means with **ε / Tγ / TΩ** lines  
  - Alive / Translated / Accepted area plots  
  - Five-Level stacked distribution (L0–L5)  
  - Ω vs γ phase plot with collapse zone  
  - Final roster with ψ/Ω/γ, κ_export, intimacy, S_old/S_new, I-point  

---

## 📊 Example Outputs

- Population means with ψ floor (ε), γ ceiling (Tγ), Ω floor (TΩ)  
- Viability & Translation area plots  
- Five-Level distribution (L0–L5)  
- Ω vs γ phase plots with collapse predicate box  
- Roster tables (ψ, Ω, γ, F, κ_export, intimacy, S_old/S_new, I-point)  

---

## 🗂️ Repository Structure
```
.
├── alphagate_app.py             # Simulator (Streamlit)
├── alphagate_dashboard.py       # Dashboard visualizer (Streamlit)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── data/                        # (optional) saved CSV logs
```

---

## 🔧 Configuration Notes

- **Collapse predicate**: ψ < ε or (γ > Tγ and Ω < TΩ)  
- **Translation gate**: κ_export ≥ threshold with non-collapse state  
- **L5 (illustrative stability)**: ψ ≥ 0.90, Ω ≥ 0.85, γ ≤ 0.10  
- **Intimacy**: running integral of F (used in L2)  
- **Trends**: γ-trend detection aids L3 classification  

---

## 💡 Tips

- Use higher **k** (faith scale) to test how regulatory drive mitigates collapse.  
- Increase drift rate/magnitude to stress systems and observe translation events.  
- Tighten export thresholds to make L4→L5 transitions rarer and more meaningful.  

---

## 🏷️ Suggested GitHub Topics

`streamlit`, `simulation`, `entropy`, `negentropy`,  
`recursive-modeling`, `symbolic-field-theory`,  
`entropic-recursion-framework`, `hyperverse`  

---

## 🌌 Inspiration

AlphaGate operationalizes concepts from:  
- Entropic Recursion Framework (ERF v3.0)  
- Symbolic Field Theory (SFT v3.0+ / v4.0)  
- Symbolic Gravity v2.1  
- Fractal Cosmic Weaver Framework v2.0  
- Observer Framework / I-Point Theory  
- Recursive Soteriology — Five-Level Model  

---

## 📜 License

MIT License — open for research, teaching, and symbolic experimentation.  

---

## ✨ Acknowledgements

Built as part of the **Symbolic Negentropy Constellation** exploration.  
Inspired by ERF, SFT, FCWF, Observer Theory, NOMAS, Symbolic Gravity, and Recursive Soteriology.
