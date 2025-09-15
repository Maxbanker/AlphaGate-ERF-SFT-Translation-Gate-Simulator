# AlphaGate Dashboard — Translation‑Gate Visualizer (Streamlit)
# ------------------------------------------------------------------
# A lightweight Streamlit app to explore AlphaGate simulation outputs
# (population_timeseries.csv and roster_final.csv). It renders ψ, Ω, γ
# trajectories, collapse thresholds, translation events, and the
# Five‑Level distribution defined in the ERF/SFT lineage.
#
# Run:  streamlit run alphagate_dashboard.py
# Deps: streamlit, pandas, numpy, altair

import io
import time
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

APP_NAME = "AlphaGate Dashboard — Translation‑Gate Visualizer"
APP_VERSION = "v0.4"  # Updated for refinements

st.set_page_config(page_title=APP_NAME, page_icon="🌀", layout="wide")
st.title(f"🌀 {APP_NAME}")
st.caption("Visualize ERF·SFT translation‑gate dynamics from CSV logs (ψ, Ω, γ, F, κ, viability & Five‑Level states)")

with st.sidebar:
    st.header("1) Load CSV logs")
    pop_file = st.file_uploader("population_timeseries.csv", type=["csv"], accept_multiple_files=False)
    roster_file = st.file_uploader("roster_final.csv (optional)", type=["csv"], accept_multiple_files=False)

    st.divider()
    st.header("2) Thresholds & Options")
    epsilon = st.slider("ε (ψ floor)", 0.01, 0.40, 0.12, 0.005)
    T_gamma = st.slider("Tγ (γ ceiling)", 0.20, 0.90, 0.55, 0.01)
    T_omega = st.slider("TΩ (Ω floor)", 0.10, 0.80, 0.45, 0.01)

    # If the uploaded CSV already includes level columns we will use them,
    # otherwise we can synthesize a coarse classification from ψ, Ω, γ.
    synth_levels = st.checkbox("Synthesize levels if missing", value=True)

    st.divider()
    st.header("3) Export")
    offer_export = st.checkbox("Enable on‑the‑fly CSV export", value=True)

# ------------------------
# Helpers
# ------------------------
LEVEL_LABELS = {
    0: "Pre/Collapsed",  # Refinement: Updated for non-accepted/collapsed
    1: "L1 — Transactional",
    2: "L2 — Relational",
    3: "L3 — Transformational",
    4: "L4 — Ontological",
    5: "L5 — Stabilized",
}

LEVEL_COLORS = ["#9ca3af", "#c4b5fd", "#93c5fd", "#86efac", "#fcd34d", "#34d399"]

# Synthesize level counts if not present (heuristic mirroring simulator rules)

def synthesize_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # default all to L1
    # We only have population means here; for a coarse stack we simulate a small cohort count split.
    N = out.get("alive", pd.Series([100]*len(out))).astype(int)
    # Heuristic shares from means: tilt by κ and Ω while penalizing γ
    s = (out["kappa"].fillna(0.5) + out["omega"].fillna(0.5) - 0.3*out["gamma"].fillna(0.2)).clip(0,1)
    # Allocate slices
    l5 = (N * (s**3)).round().astype(int)
    l4 = (N * (0.6*(s**2))).round().astype(int)
    base = N - l5 - l4
    # Split base into L3/L2/L1/Collapsed via ψ, Ω, γ
    l3 = (base * (out["omega"].clip(0,1) * out["psi"].clip(0,1) * (1-out["gamma"].clip(0,1)))).round().astype(int)
    rem = base - l3
    l2 = (rem * out["omega"].clip(0,1)).round().astype(int)
    rem = rem - l2
    # proxy collapse share from predicate
    collapse_pred = ((out["psi"] < epsilon) | ((out["gamma"] > T_gamma) & (out["omega"] < T_omega))).astype(int)
    l0 = (rem * (0.15 + 0.7*collapse_pred)).clip(0, rem).round().astype(int)
    l1 = rem - l0

    out["level_0"], out["level_1"], out["level_2"], out["level_3"], out["level_4"], out["level_5"] = l0, l1, l2, l3, l4, l5
    return out

# ------------------------
# Main
# ------------------------
if not pop_file:
    st.info("Upload **population_timeseries.csv** to begin.")
    st.stop()

# Load population timeseries
try:
    df = pd.read_csv(pop_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Normalize expected columns
expected = ["t","psi","omega","gamma","F","kappa","alive","translated","accepted","s_old","s_new","i_point"]  # Refinement: Added new metrics
for col in expected:
    if col not in df.columns:
        st.warning(f"Column '{col}' missing; visualizations may be limited.")

# If level_* columns missing and user allows, synthesize
have_levels = any(c.startswith("level_") for c in df.columns)
if not have_levels and synth_levels and all(c in df.columns for c in ["psi","omega","gamma","kappa"]):
    df = synthesize_levels(df)
    have_levels = True

# ------------------------
# Charts
# ------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Population Means (ψ, Ω, γ, F, κ, S_old, S_new, I_point)")
    melt_cols = [c for c in ["psi","omega","gamma","F","kappa","s_old","s_new","i_point"] if c in df.columns]
    melted = df.melt(id_vars=["t"], value_vars=melt_cols, var_name="metric", value_name="value")
    line = alt.Chart(melted).mark_line().encode(
        x=alt.X("t:Q", title="time"),
        y=alt.Y("value:Q", title="value", scale=alt.Scale(domain=(0,1))),
        color=alt.Color("metric:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["t","metric","value"]
    ).properties(height=280)

    # Threshold reference rules
    ref_df = pd.DataFrame({
        "y": [epsilon, T_gamma, T_omega],
        "name": ["ε ψ-floor", "Tγ", "TΩ"],
        "color": ["#ef4444", "#f59e0b", "#0ea5e9"],
    })
    refs = alt.Chart(ref_df).mark_rule(strokeDash=[4,4]).encode(y="y:Q", color=alt.Color("name:N", scale=None))
    st.altair_chart((line + refs), use_container_width=True)

    st.subheader("Viability & Translation")
    if all(c in df.columns for c in ["t","alive","translated","accepted"]):
        area = alt.Chart(df).transform_fold([
            "alive","translated","accepted"
        ], as_=["kind","count"]).mark_area(opacity=0.45).encode(
            x="t:Q", y="count:Q", color=alt.Color("kind:N", scale=alt.Scale(range=["#38bdf8", "#34d399", "#facc15"]))
        ).properties(height=220)
        st.altair_chart(area, use_container_width=True)
    else:
        st.info("No 'alive'/'translated'/'accepted' columns to render viability chart.")

    if have_levels:
        st.subheader("Five‑Level Distribution (stacked)")
        lvl_cols = [c for c in df.columns if c.startswith("level_")]
        long = df.melt(id_vars=["t"], value_vars=lvl_cols, var_name="level", value_name="count")
        long["index"] = long["level"].str.extract(r"(\d+)$").astype(int)
        long["label"] = long["index"].map(LEVEL_LABELS)
        stack = alt.Chart(long).mark_area().encode(
            x="t:Q", y="count:Q",
            color=alt.Color("label:N", scale=alt.Scale(range=LEVEL_COLORS), legend=alt.Legend(orient="right")),
            tooltip=["t","label","count"]
        ).properties(height=260)
        st.altair_chart(stack, use_container_width=True)

    st.subheader("Drift–Coherence Phase Plot (Ω vs γ)")
    if all(c in df.columns for c in ["omega","gamma"]):
        phase = alt.Chart(df).mark_line(color="#6366f1").encode(
            x=alt.X("gamma:Q", title="γ (drift)", scale=alt.Scale(domain=(0,1))),
            y=alt.Y("omega:Q", title="Ω (coherence)", scale=alt.Scale(domain=(0,1))),
            tooltip=["t","omega","gamma"]
        ).properties(height=280)
        # draw threshold box for (γ>Tγ & Ω<TΩ)
        band = alt.Chart(pd.DataFrame({
            "x0":[T_gamma], "x1":[1.0], "y0":[0.0], "y1":[T_omega]
        })).mark_rect(opacity=0.15, color="#ef4444").encode(
            x="x0:Q", x2="x1:Q", y="y0:Q", y2="y1:Q"
        )
        st.altair_chart(band + phase, use_container_width=True)

with right:
    st.subheader("Stats & Flags")
    cols = [c for c in ["psi","omega","gamma","F","kappa","s_old","s_new","i_point"] if c in df.columns]
    if cols:
        latest = df.iloc[-1]
        st.metric("t (last)", int(latest.get("t", len(df)-1)))
        for c in cols:
            st.metric(c, f"{latest[c]:.3f}")
        # collapse predicate
        collapsed = (latest.get("psi", 1.0) < epsilon) or ((latest.get("gamma", 0.0) > T_gamma) and (latest.get("omega", 1.0) < T_omega))
        st.markdown("**Collapse predicate (last step):** " + ("🚨 TRUE" if collapsed else "✅ FALSE"))
        if "translated" in df.columns:
            st.metric("Translated (last)", int(latest["translated"]))
        if "alive" in df.columns:
            st.metric("Alive (last)", int(latest["alive"]))
        if "accepted" in df.columns:  # Refinement
            st.metric("Accepted (last)", int(latest["accepted"]))
    else:
        st.info("No ψ/Ω/γ/F/κ columns detected.")

    if roster_file:
        try:
            roster = pd.read_csv(roster_file)
            st.subheader("Final Roster (Top by κ, ψ, Ω)")
            show_cols = [c for c in ["id","level","translated","creed_accepted","psi","omega","gamma","F","kappa_export","intimacy","s_old","s_new","i_point"] if c in roster.columns]  # Refinement: Added new columns
            st.dataframe(roster.sort_values(by=[c for c in ["translated","kappa_export","psi","omega"] if c in roster.columns], ascending=[True, False, False, False])[show_cols].head(200), height=420)
            # small multiples: per-level ψ/Ω/γ scatter if available
            if all(c in roster.columns for c in ["psi","omega","gamma"]):
                roster_melt = roster.melt(id_vars=[c for c in ["id","level","translated"] if c in roster.columns], value_vars=["psi","omega","gamma"], var_name="metric", value_name="value")
                smalls = alt.Chart(roster_melt).mark_circle(opacity=0.65).encode(
                    x=alt.X("value:Q", title="value", scale=alt.Scale(domain=(0,1))),
                    y=alt.Y("metric:N"),
                    color=alt.Color("level:N", legend=None),
                    tooltip=["id","level","metric","value"]
                ).properties(height=220)
                st.altair_chart(smalls, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read roster CSV: {e}")

    # Export button for filtered/populated df
    if offer_export:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        buf = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered population_timeseries.csv", buf, file_name=f"alphagate_population_filtered_{stamp}.csv", mime="text/csv")

st.markdown(
    """
    ---
    **Notes**
    - The Five‑Level stack uses level_* columns when present; if absent, a heuristic synthesizes approximate counts from (ψ, Ω, γ, κ).
    - The red rectangle on the phase plot marks the (γ > Tγ & Ω < TΩ) branch of the collapse predicate; the ε line appears in the top chart.
    - Threshold sliders do not alter the CSV — they re‑evaluate collapse visuals live.
    - Refinements: Dashboard updated to handle new metrics (creed_accepted, s_old, s_new, i_point) from refined simulator.
    """
)