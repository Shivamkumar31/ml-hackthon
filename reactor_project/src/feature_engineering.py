"""
Physics-informed feature engineering for the reactor yield prediction problem.

Reaction network:   A --k1--> B --k2--> C   (series-parallel, non-isothermal)

We can't know the true Ea/A (pre-exponential) values, so instead of computing
literal rate constants we build DIMENSIONLESS / PHYSICALLY-MOTIVATED proxy
features that a tree-based or kernel model can combine to approximate the
true (unknown) Arrhenius kinetics. This is the standard "surrogate modeling"
trick used in chemical engineering ML: you don't need the exact mechanistic
equation, you need features that carry the same physical information.
"""

import numpy as np
import pandas as pd

R_GAS = 8.314  # J/mol/K, universal gas constant


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- 1. Residence time proxy -------------------------------------
    # True tau = Volume / flow_rate. We don't have cross-sectional area,
    # so length_m is a monotonic proxy for volume (constant-diameter tube
    # assumption, standard for flow reactors).
    df["residence_time_proxy"] = df["length_m"] / df["flow_rate_L_min"]

    # ---- 2. Arrhenius-style inverse-temperature terms -----------------
    # k = A * exp(-Ea / (R*T))  ->  ln(k) is linear in 1/T.
    # We give the model 1/T directly so it can learn an effective
    # exp(-Ea/RT) relationship without us knowing Ea.
    df["inv_inlet_T"] = 1.0 / df["inlet_temperature_K"]
    df["inv_jacket_T"] = 1.0 / df["jacket_temperature_K"]

    # ---- 3. Heat transfer driving force --------------------------------
    # jacket vs inlet temperature gap tells us whether the reactor is being
    # heated or cooled relative to feed -> drives the *actual* in-reactor
    # temperature profile, not just the feed temperature.
    df["jacket_inlet_deltaT"] = df["jacket_temperature_K"] - df["inlet_temperature_K"]
    df["avg_thermal_T"] = (df["inlet_temperature_K"] + df["jacket_temperature_K"]) / 2.0

    # ---- 4. Damkohler-like number (dimensionless reaction extent) ----
    # Da = k * tau. We proxy "k" with an Arrhenius-shaped term using the
    # average operating temperature, scaled arbitrarily (the model will
    # learn the right scale via its own weights/splits).
    df["damkohler_proxy"] = df["residence_time_proxy"] * np.exp(
        -5000.0 / (R_GAS * df["avg_thermal_T"])
    )

    # ---- 5. Throughput / loading ----------------------------------------
    df["throughput"] = df["flow_rate_L_min"] * df["concentration_mol_L"]

    # ---- 6. Non-monotonic yield proxy: tau vs temperature interaction --
    # Captures the "too short tau -> A doesn't convert; too long tau at
    # high T -> B over-reacts to C" trade-off that defines a yield optimum.
    df["tau_times_T"] = df["residence_time_proxy"] * df["avg_thermal_T"]
    df["tau_squared"] = df["residence_time_proxy"] ** 2  # curvature term

    # ---- 7. Concentration-temperature coupling --------------------------
    df["conc_times_invT"] = df["concentration_mol_L"] * df["inv_inlet_T"]

    return df


FEATURE_COLUMNS = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
    "residence_time_proxy",
    "inv_inlet_T",
    "inv_jacket_T",
    "jacket_inlet_deltaT",
    "avg_thermal_T",
    "damkohler_proxy",
    "throughput",
    "tau_times_T",
    "tau_squared",
    "conc_times_invT",
]
