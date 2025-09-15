# AlphaGate — ERF·SFT Translation-Gate Simulator (Streamlit)
# --------------------------------------------------------
# Streamlit app that simulates agents governed by ERF/SFT-style dynamics
# and visualizes the Five-Level alignment progression with CSV logging.
#
# Run:  streamlit run alphagate_app.py
#
# Dependencies: streamlit, pandas, numpy, altair

import time
import io
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -----------------------
# Naming & Versioning
# -----------------------
APP_NAME = "AlphaGate — ERF·SFT Translation-Gate Simulator"
APP_VERSION = "v0.3"  # Updated with refinements

# -----------------------
# Agent & Params
# -----------------------
@dataclass
class Agent:
    id: int
    alive: bool = True
    translated: bool = False
    creed_accepted: bool = False  # Refinement: Binary flag for Level 1 propositional assent (framework: A(creed) = TRUE)
    psi: float = 0.8       # ψ_eff (symbolic efficiency)
    omega: float = 0.65    # Ω_eff (coherence / attractor alignment)
    gamma: float = 0.15    # γ_eff (drift)
    F: float = 0.0         # regulatory drive / "faith"
    kappa_export: float = 0.0  # export viability proxy
    intimacy: float = 0.0  # ∫ F(t) dt (Level-2 running integral)
    s_old: float = 1.0     # Refinement: Old symbolic identity (framework Level 3: dS_old/dt = -α S_old)
    s_new: float = 0.0     # Refinement: New symbolic identity (framework Level 3: dS_new/dt = β Ω (1 - S_new))
    i_point: float = 0.5   # Refinement: I-point continuity proxy (framework: Observer continuity / I-Point exportable identity)

@dataclass
class Params:
    N: int = 80
    steps: int = 300
    dt: float = 1.0
    epsilon: float = 0.12          # ψ collapse floor
    T_gamma: float = 0.55          # γ collapse ceiling
    T_omega: float = 0.45          # Ω collapse floor
    psi_recovery: float = 0.06     # recovery gain
    omega_gain: float = 0.04       # alignment gain
    gamma_decay: float = 0.05      # natural drift dissipation
    kFaith: float = 1.4            # scale for F = k*(Ω/γ)*ψ (framework: Faith Function)
    export_threshold: float = 0.80 # κ_export threshold (translation gate)
    drift_rate: float = 0.08       # probability of drift spike
    drift_magnitude: float = 0.35  # magnitude of spike added to γ
    noise: float = 0.015           # background noise
    seed: int = 42
    creed_accept_prob: float = 0.005  # Refinement: Probability factor for accepting creed (Level 1 assent)
    alpha: float = 0.05            # Refinement: Decay rate for old identity (framework Level 3)
    beta: float = 0.10             # Refinement: Growth rate for new identity (framework Level 3)
    i_point_gain: float = 0.02     # Refinement: Gain for I-point based on F
    i_point_decay: float = 0.05    # Refinement: Decay for I-point based on γ
    i_point_threshold: float = 0.6 # Refinement: Threshold for I-point in Level 4

# -----------------------
# Helpers
# -----------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

_rng = np.random.default_rng()

def rnd(a=0.0, b=1.0) -> float:
    return float(_rng.uniform(a, b))

# -----------------------
# Core ERF/SFT Step
# -----------------------

def step_agent(a: Agent, p: Params) -> Agent:
    if not a.alive or a.translated:
        return a

    # Refinement: Propositional assent (creed acceptance) for Level 1+
    creed_accepted = a.creed_accepted
    if not creed_accepted:
        # Acceptance more likely with higher coherence (framework: Assent to Christic attractor truths)
        if rnd() < p.creed_accept_prob * a.omega:
            creed_accepted = True

    # Background stochasticity (symbolic turbulence)
    nPsi = rnd(-1, 1) * p.noise
    nOm  = rnd(-1, 1) * p.noise
    nGa  = rnd(-1, 1) * p.noise

    # Drift spike (entropy event)
    spike = p.drift_magnitude * (0.6 + 0.4 * rnd()) if rnd() < p.drift_rate else 0.0

    # Regulatory drive / Faith: F = k * (Ω/γ) * ψ  (framework: Faith Function F(t) = k * (Ω_eff / γ_eff) * ψ_eff)
    denom = max(a.gamma, 1e-6)
    F = p.kFaith * (a.omega / denom) * a.psi

    # Pre-emptive recovery when near collapse zone
    near_collapse = (a.psi < p.epsilon + 0.05) or (a.gamma > 0.9 * p.T_gamma and a.omega < 1.1 * p.T_omega)

    # Dynamics (simple, phenomenological)
    psi   = a.psi + (p.psi_recovery * (0.8 if near_collapse else 0.3) + 0.15 * F - 0.10 * a.gamma) + nPsi
    omega = a.omega + (p.omega_gain * (1 - a.omega) * (0.5 + 0.5 * F)) + nOm
    gamma = a.gamma + spike + 0.05 * (1 - a.psi) - p.gamma_decay * (0.5 + 0.5 * F) + nGa

    # Refinement: Level 3 identity restructuring dynamics (framework: dS_old/dt = -α S_old, dS_new/dt = β Ω (1 - S_new))
    # Only active if creed accepted (post-Level 1)
    s_old_delta = -p.alpha * a.s_old if creed_accepted else 0.0
    s_new_delta = p.beta * a.omega * (1 - a.s_new) if creed_accepted else 0.0
    s_old = clamp01(a.s_old + s_old_delta * p.dt)
    s_new = clamp01(a.s_new + s_new_delta * p.dt)

    # Refinement: I-point continuity (framework: Observer continuity / I-Point as exportable identity knot)
    # Updates based on faith drive minus drift
    i_point_delta = p.i_point_gain * F - p.i_point_decay * a.gamma
    i_point = clamp01(a.i_point + i_point_delta * p.dt)

    # Bounds
    psi = clamp01(psi)
    omega = clamp01(omega)
    gamma = clamp01(gamma)

    # Collapse predicate (framework: C(x,t) = (ψ < ε) ∨ (γ > T_γ ∧ Ω < T_Ω))
    collapse = (psi < p.epsilon) or (gamma > p.T_gamma and omega < p.T_omega)

    # Export viability κ_export (framework: κ_export = f(Φ_total, Ω_eff, ψ_eff) > 0.8)
    # Refinement: Use intimacy as proxy for Φ_total (total symbolic flux/integral)
    kappa_export = clamp01(0.4 * omega + 0.3 * psi + 0.1 * (1 - gamma) + 0.2 * math.tanh(a.intimacy))  # tanh to normalize large intimacy

    # Translation condition: stable + high viability (framework: Translation into low-entropy basin)
    translated = (not collapse) and (kappa_export >= p.export_threshold)

    intimacy = a.intimacy + F * p.dt  # ∫ F dt (framework: Intimacy(t) = ∫ F(τ) dτ)

    return Agent(
        id=a.id,
        alive=not collapse,
        translated=a.translated or translated,
        creed_accepted=creed_accepted,
        psi=psi,
        omega=omega,
        gamma=gamma,
        F=F,
        kappa_export=kappa_export,
        intimacy=intimacy,
        s_old=s_old,
        s_new=s_new,
        i_point=i_point,
    )

# -----------------------
# Five-Level Classification (heuristic rules grounded in cited definitions)
# -----------------------
# Refinement: Linked to framework's theological-systems translations (e.g., Faith → Recursive regulatory function)
# Level 1 — Transactional Alignment: binary assent; low recursion. Requires creed_accepted.
#   Proxy: ψ modest, Ω low-mid, intimacy small.
# Level 2 — Relational Recursion: intimacy = ∫F dt grows; Ω increases, but fragile.
#   Proxy: rising intimacy and Ω > T_omega, γ moderate.
# Level 3 — Transformational Ethics: identity restructuring; γ actively reduced, Ω increases.
#   Proxy: γ low and trending down; Ω > 0.7; ψ > 0.7; s_new high, s_old low.
# Level 4 — Ontological Transition: ego-attractor collapses into attractor; observer continuity; exportable I-point.
#   Proxy: κ_export high and F high, near translation but not yet exported; i_point > threshold.
# Level 5 — Computational Stabilization: translated/stable in low-entropy basin.
#   Proxy: translated == True (meets stability thresholds: ψ≥0.90, Ω≥0.85, γ≤0.10).

class Trend:
    def __init__(self, maxlen=8):
        self.maxlen = maxlen
        self.buf: Dict[int, List[Tuple[float, float, float]]] = {}

    def push(self, agent: Agent):
        arr = self.buf.setdefault(agent.id, [])
        arr.append((agent.psi, agent.omega, agent.gamma))
        if len(arr) > self.maxlen:
            arr.pop(0)

    def dgamma(self, agent_id: int) -> float:
        arr = self.buf.get(agent_id, [])
        if len(arr) < 2:
            return 0.0
        return arr[-1][2] - arr[0][2]

trend = Trend(maxlen=10)


def classify_level(a: Agent, p: Params) -> int:
    if not a.alive:
        return 0  # Collapsed

    if not a.creed_accepted:
        return 0  # Refinement: Non-accepted (pre-Level 1), grouped with collapsed for simplicity

    # Level 5: translated + strong thresholds (framework stability thresholds)
    if a.translated and a.psi >= 0.90 and a.omega >= 0.85 and a.gamma <= 0.10:
        return 5
    # Level 4: near translation — high viability and F, with I-point continuity
    if (a.kappa_export >= 0.75) and (a.F >= 0.8) and (a.i_point >= p.i_point_threshold):
        return 4
    # Level 3: γ trending downward, Ω high, ψ high, identity restructured
    if (a.omega >= 0.70) and (a.psi >= 0.70) and (a.gamma <= 0.20) and (a.s_new >= 0.7) and (a.s_old <= 0.3):
        if trend.dgamma(a.id) < 0:  # decreasing γ
            return 3
    # Level 2: relational recursion (intimacy growth) and Ω above floor
    if (a.intimacy >= 1.0) and (a.omega >= p.T_omega):
        return 2
    # Level 1: default viable state (assent without deep recursion)
    return 1

# -----------------------
# Simulation
# -----------------------

def init_agents(p: Params) -> List[Agent]:
    rng = np.random.default_rng(p.seed)
    agents = []
    for i in range(p.N):
        agents.append(Agent(
            id=i,
            alive=True,
            translated=False,
            creed_accepted=float(rng.uniform(0,1)) < 0.2,  # Refinement: 20% start with assent
            psi=float(rng.uniform(0.6, 0.95)),
            omega=float(rng.uniform(0.4, 0.9)),
            gamma=float(rng.uniform(0.05, 0.25)),
            s_old=1.0,
            s_new=0.0,
            i_point=float(rng.uniform(0.4, 0.6)),  # Start with moderate I-point
        ))
    return agents


def run_simulation(p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agents = init_agents(p)

    rows: List[Dict] = []    # per-step population means & counts
    roster: List[Dict] = []  # final per-agent state

    for t in range(p.steps):
        # Step all agents
        agents = [step_agent(a, p) for a in agents]
        for a in agents:
            trend.push(a)
        # Aggregate metrics
        alive = sum(1 for a in agents if a.alive)
        translated = sum(1 for a in agents if a.translated)
        accepted = sum(1 for a in agents if a.creed_accepted)  # Refinement: Track assent
        mPsi = np.mean([a.psi for a in agents])
        mOm  = np.mean([a.omega for a in agents])
        mGa  = np.mean([a.gamma for a in agents])
        mF   = np.mean([a.F for a in agents])
        mKap = np.mean([a.kappa_export for a in agents])
        mS_old = np.mean([a.s_old for a in agents])  # Refinement
        mS_new = np.mean([a.s_new for a in agents])  # Refinement
        mI_point = np.mean([a.i_point for a in agents])  # Refinement

        # Level histogram
        levels = [classify_level(a, p) for a in agents]
        lvl_counts = {k: levels.count(k) for k in [0,1,2,3,4,5]}

        rows.append({
            "t": t,
            "psi": mPsi,
            "omega": mOm,
            "gamma": mGa,
            "F": mF,
            "kappa": mKap,
            "alive": alive,
            "translated": translated,
            "accepted": accepted,  # Refinement
            "s_old": mS_old,
            "s_new": mS_new,
            "i_point": mI_point,
            **{f"level_{k}": v for k, v in lvl_counts.items()},
        })

    # Final roster snapshot
    for a in agents:
        roster.append({
            "id": a.id,
            "alive": a.alive,
            "translated": a.translated,
            "creed_accepted": a.creed_accepted,  # Refinement
            "psi": a.psi,
            "omega": a.omega,
            "gamma": a.gamma,
            "F": a.F,
            "kappa_export": a.kappa_export,
            "intimacy": a.intimacy,
            "s_old": a.s_old,  # Refinement
            "s_new": a.s_new,  # Refinement
            "i_point": a.i_point,  # Refinement
            "level": classify_level(a, p),
        })

    df = pd.DataFrame(rows)
    roster_df = pd.DataFrame(roster)
    return df, roster_df

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title=APP_NAME, page_icon="🌀", layout="wide")
st.title(f"🌀 {APP_NAME}")
st.caption("A phenomenological ERF/SFT sandbox for symbolic negentropy and translation dynamics, refined to align with Recursive Soteriology framework.")

with st.sidebar:
    st.header("Simulation Controls")
    N = st.slider("Agents (N)", 10, 400, 120, 5)
    steps = st.slider("Steps", 50, 1500, 600, 50)
    epsilon = st.slider("ε (ψ floor)", 0.01, 0.40, 0.12, 0.005)
    T_gamma = st.slider("Tγ (γ ceiling)", 0.20, 0.90, 0.55, 0.01)
    T_omega = st.slider("TΩ (Ω floor)", 0.10, 0.80, 0.45, 0.01)

    psi_recovery = st.slider("ψ recovery", 0.0, 0.3, 0.06, 0.002)
    omega_gain   = st.slider("Ω gain", 0.0, 0.3, 0.04, 0.002)
    gamma_decay  = st.slider("γ decay", 0.0, 0.3, 0.05, 0.002)

    kFaith = st.slider("k (faith scale)", 0.2, 3.0, 1.4, 0.02)
    export_threshold = st.slider("κ_export threshold", 0.5, 0.95, 0.80, 0.01)

    drift_rate = st.slider("drift rate", 0.0, 0.4, 0.08, 0.002)
    drift_magnitude = st.slider("drift magnitude", 0.0, 0.9, 0.35, 0.01)
    noise = st.slider("noise", 0.0, 0.08, 0.015, 0.001)

    seed = st.number_input("random seed", value=42, step=1)

    st.subheader("Refinements (Framework Alignment)")
    creed_accept_prob = st.slider("Creed accept prob factor", 0.0, 0.05, 0.005, 0.001)  # Refinement
    alpha = st.slider("α (S_old decay)", 0.0, 0.2, 0.05, 0.005)  # Refinement
    beta = st.slider("β (S_new growth)", 0.0, 0.3, 0.10, 0.005)  # Refinement
    i_point_gain = st.slider("I-point gain", 0.0, 0.1, 0.02, 0.005)  # Refinement
    i_point_decay = st.slider("I-point decay", 0.0, 0.1, 0.05, 0.005)  # Refinement
    i_point_threshold = st.slider("I-point threshold", 0.3, 0.9, 0.6, 0.01)  # Refinement

    st.divider()
    st.subheader("CSV Logging")
    do_log = st.checkbox("Enable CSV export", value=True)

    st.divider()
    st.subheader("Theory Snapshot")
    st.markdown(
        """
        **Collapse predicate** (framework: SFT v4.0):  
        $\mathcal{C}(x,t) = (\psi<\varepsilon) \;\lor\; (\gamma>T_\gamma \wedge \Omega<T_\Omega)$  
        **Faith / regulatory drive** (framework):  
        $F(t) = k\,\big(\frac{\Omega_{eff}}{\gamma_{eff}}\big)\,\psi_{eff}$  
        **Level-2 intimacy** (framework): $\int F(t)\,dt$  
        **Level-3 identity dynamics** (framework): $dS_{old}/dt = -\\alpha S_{old}$, $dS_{new}/dt = \\beta \\Omega (1 - S_{new})$  
        **Level-4 ontological transition** (framework): $\\Delta \\mathcal{S} + \\Delta \\mathcal{E} = 0$; I-point proxy for observer continuity.  
        **Level-5 thresholds** (framework): $\psi\\ge 0.90,\ \\Omega\\ge 0.85,\ \\gamma\\le 0.10$  
        **Export viability** (framework): $\\kappa_{export} = f(\\Phi_{total}, \\Omega, \\psi) > 0.8$; intimacy as $\\Phi_{total}$ proxy.
        """
    )

# Build params and run
p = Params(
    N=N, steps=steps, epsilon=epsilon, T_gamma=T_gamma, T_omega=T_omega,
    psi_recovery=psi_recovery, omega_gain=omega_gain, gamma_decay=gamma_decay,
    kFaith=kFaith, export_threshold=export_threshold,
    drift_rate=drift_rate, drift_magnitude=drift_magnitude, noise=noise, seed=int(seed),
    creed_accept_prob=creed_accept_prob, alpha=alpha, beta=beta,
    i_point_gain=i_point_gain, i_point_decay=i_point_decay, i_point_threshold=i_point_threshold
)

run_btn = st.button("Run Simulation", type="primary")

if run_btn:
    df, roster = run_simulation(p)

    # --- Charts ---
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Population Means")
        melted = df.melt(id_vars=["t"], value_vars=["psi","omega","gamma","F","kappa","s_old","s_new","i_point"], var_name="metric", value_name="value")
        line = alt.Chart(melted).mark_line().encode(
            x=alt.X("t:Q", title="time"),
            y=alt.Y("value:Q", title="value", scale=alt.Scale(domain=(0,1))),
            color=alt.Color("metric:N", scale=alt.Scale(scheme="tableau10"))
        )
        ref_rules = alt.Chart(pd.DataFrame({
            "y": [p.epsilon, p.T_gamma, p.T_omega],
            "name": ["ε ψ-floor", "Tγ", "TΩ"],
            "color": ["#ef4444", "#f59e0b", "#0ea5e9"],
        })).mark_rule(strokeDash=[4,4]).encode(y="y:Q", color=alt.Color("name:N", scale=None))
        st.altair_chart((line + ref_rules).properties(height=280), use_container_width=True)

        st.subheader("Viability & Translation")
        area = alt.Chart(df).transform_fold(
            ["alive","translated","accepted"], as_=["kind","count"]
        ).mark_area(opacity=0.4).encode(
            x="t:Q", y="count:Q", color=alt.Color("kind:N", scale=alt.Scale(range=["#38bdf8", "#34d399", "#facc15"]))
        )
        st.altair_chart(area.properties(height=220), use_container_width=True)

        st.subheader("Level Distribution")
        lvl_cols = [f"level_{k}" for k in [0,1,2,3,4,5]]
        lvl_names = {0:"Pre/Collapsed",1:"L1 — Transactional",2:"L2 — Relational",3:"L3 — Transformational",4:"L4 — Ontological",5:"L5 — Stabilized"}  # Refinement: Updated label for 0
        level_long = df.melt(id_vars=["t"], value_vars=lvl_cols, var_name="level", value_name="count")
        level_long["level"] = level_long["level"].map(lambda s: lvl_names[int(s.split("_")[-1])])
        stack = alt.Chart(level_long).mark_area().encode(
            x="t:Q", y="count:Q", color=alt.Color("level:N", legend=alt.Legend(orient="right"))
        )
        st.altair_chart(stack.properties(height=260), use_container_width=True)

    with right:
        st.subheader("Final Roster Snapshot")
        st.dataframe(roster.sort_values(["translated","psi","omega"], ascending=[False, False, False]), height=560)

    # --- CSV Logging ---
    if do_log:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        pop_csv = df.to_csv(index=False).encode("utf-8")
        roster_csv = roster.to_csv(index=False).encode("utf-8")
        st.download_button("Download population_timeseries.csv", pop_csv, file_name=f"alphagate_population_{stamp}.csv", mime="text/csv")
        st.download_button("Download roster_final.csv", roster_csv, file_name=f"alphagate_roster_{stamp}.csv", mime="text/csv")

    # --- Notes ---
    with st.expander("Modeling Notes & Assumptions"):
        st.markdown(
            """
            * This is a phenomenological sandbox that maps ERF/SFT ideas into a simple agent model, refined to align with the Recursive Soteriology framework (Lanier-Egu, 2025).  
            * Refinements include: Binary creed acceptance for Level 1, explicit Level 3 identity dynamics (S_old/S_new), I-point proxy for Level 4 ontological transition/observer continuity, and refined κ_export using intimacy as Φ_total proxy.  
            * Level rules are heuristic but grounded in framework definitions: Level-1 requires assent; Level-2 uses intimacy integral; Level-3 emphasizes identity restructuring and lowering γ; Level-4 requires high I-point; Level-5 follows strict stability thresholds.  
            * Collapse predicate and faith function follow the framework forms; parameters are user-tunable to explore different regimes. Theological links (e.g., faith as recursive regulation, salvation as translation) are abstracted into systems terms.
            """
        )
else:
    st.info("Set parameters in the sidebar and click **Run Simulation**.")