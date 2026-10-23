#!/usr/bin/env python3
"""Revised DFT–SPI–ML analysis for the 76-system melphalan dataset.

Place this script beside:
  1. only table.xlsx          (raw DFT descriptors; authoritative input)
  2. Rank_Shift_Table.xlsx    (old published/draft ranks; comparison only)

Run in Google Colab or locally:
    python revised_spi_reviewer_analysis.py

Optional packages for XGBoost and SHAP:
    pip install xgboost shap

Scientific interpretation:
* The ML task is surrogate reconstruction of a constructed SPI target.
* R2/MAE/RMSE are not experimental sensor-accuracy measures.
* The inverse-band-gap comparison is exploratory electronic accessibility,
  not independent validation of adsorption-induced sensor response.
"""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
RAW_FILE = Path("only table.xlsx")
OLD_RANK_FILE = Path("Rank_Shift_Table.xlsx")
OUTPUT_DIR = Path("revised_spi_results")
RANDOM_STATE = 42
N_MONTE_CARLO = 10_000
TOP_K = 5

# Recovery-time convention used consistently for every system.
K_B_EV_PER_K = 8.617333262e-5
TEMPERATURE_K = 298.15
ATTEMPT_FREQUENCY_S = 1.0e12

# Relative/absolute perturbations for ranking uncertainty. State these assumed
# uncertainty ranges explicitly in the paper; change them if you have a more
# appropriate benchmark for your DFT protocol.
EADS_RELATIVE_SD = 0.05       # 5% relative SD
CHARGE_RELATIVE_SD = 0.10     # 10% relative SD
LOG10_TAU_ABSOLUTE_SD = 0.25  # ±0.25 log10 units (1 SD)

# Exploratory adsorption desirability window. This is an alternative robustness
# test, not a universally validated physical law. Report all four boundaries.
ADS_ZERO_LOW = 0.00
ADS_IDEAL_LOW = 0.20
ADS_IDEAL_HIGH = 1.00
ADS_ZERO_HIGH = 2.00

REQUIRED_COLUMNS = [
    "System_ID", "Nanostructure_Type", "Medium", "Distance_A",
    "HOMO_eV", "LUMO_eV", "Bandgap_eV", "Charge_Transfer_e",
    "E_ads_DFT_eV", "Recoverytime_sec",
]

SCENARIOS = {
    "Baseline_40_40_20": (0.40, 0.40, 0.20),
    "Equal_33_33_33": (1/3, 1/3, 1/3),
    "Adsorption_50_30_20": (0.50, 0.30, 0.20),
    "Adsorption_60_20_20": (0.60, 0.20, 0.20),
    "Charge_30_50_20": (0.30, 0.50, 0.20),
    "Charge_20_60_20": (0.20, 0.60, 0.20),
    "Recovery_30_30_40": (0.30, 0.30, 0.40),
    "Recovery_20_20_60": (0.20, 0.20, 0.60),
    "Balanced_45_45_10": (0.45, 0.45, 0.10),
}


def minmax(series):
    """Min–max scaling with transparent handling of missing/constant data."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = x.min(), x.max()
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(np.nan, index=x.index)
    if np.isclose(lo, hi):
        return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)


def desirability_window(x, zero_low, ideal_low, ideal_high, zero_high):
    """Trapezoidal score: low/very strong binding penalized, ideal window = 1."""
    x = pd.to_numeric(x, errors="coerce").astype(float).to_numpy()
    d = np.zeros_like(x, dtype=float)
    rising = (x > zero_low) & (x < ideal_low)
    d[rising] = (x[rising] - zero_low) / (ideal_low - zero_low)
    d[(x >= ideal_low) & (x <= ideal_high)] = 1.0
    falling = (x > ideal_high) & (x < zero_high)
    d[falling] = (zero_high - x[falling]) / (zero_high - ideal_high)
    d[np.isnan(x)] = np.nan
    return pd.Series(d)


def infer_family(value):
    s = str(value).lower()
    if "fluorographene" in s:
        return "Fluorographene"
    if "graphene" in s:
        return "Graphene"
    if "bcnns" in s or "bcn sheet" in s:
        return "BNCNS"
    if "bnns" in s or "bn sheet" in s:
        return "BNNS"
    if "cage" in s or "bncnc" in s:
        return "BNC cage"
    return "Other"


def prepare_raw_data(path):
    df = pd.read_excel(path, sheet_name=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "Distance_A", "HOMO_eV", "LUMO_eV", "Bandgap_eV",
        "Mulliken_Charge_Dopant", "Charge_Transfer_e", "E_ads_DFT_eV",
        "Recoverytime_sec", "electronegativity_eV", "chemical potential_eV",
        "hardness_eV", "electrophilicity index_eV",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recalculate missing band gaps directly from orbital energies.
    computed_gap = df["LUMO_eV"] - df["HOMO_eV"]
    df["Bandgap_eV"] = df["Bandgap_eV"].fillna(computed_gap)
    # Preserve heterogeneous/reported recovery times for auditing, then create
    # one internally consistent recovery column from |Eads|. The recalculated
    # values, rather than the reported values, enter every revised SPI.
    df["Recoverytime_reported_sec"] = df["Recoverytime_sec"]
    df["Recoverytime_recalculated_sec"] = (
        1.0 / ATTEMPT_FREQUENCY_S
    ) * np.exp(
        df["E_ads_DFT_eV"].abs() / (K_B_EV_PER_K * TEMPERATURE_K)
    )
    df["Recoverytime_sec"] = df["Recoverytime_recalculated_sec"]

    df["Medium_clean"] = df["Medium"].astype(str).str.strip().str.lower()
    df["Candidate_Key"] = df["System_ID"].astype(str).str.strip() + " | " + df["Medium_clean"]
    df["Family"] = df["Nanostructure_Type"].map(infer_family)

    # Correct physical transformations stated in the revised Methods.
    df["Abs_Eads_eV"] = df["E_ads_DFT_eV"].abs()
    df["Abs_Charge_Transfer_e"] = df["Charge_Transfer_e"].abs()
    df["log10_Recovery_s"] = np.log10(df["Recoverytime_sec"].where(df["Recoverytime_sec"] > 0))
    df["A_affinity"] = minmax(df["Abs_Eads_eV"])
    df["C_charge"] = minmax(df["Abs_Charge_Transfer_e"])
    df["R_log_recovery"] = 1.0 - minmax(df["log10_Recovery_s"])
    df["Electronic_accessibility"] = minmax(1.0 / df["Bandgap_eV"].where(df["Bandgap_eV"] > 0))
    df["A_window"] = desirability_window(
        df["Abs_Eads_eV"], ADS_ZERO_LOW, ADS_IDEAL_LOW,
        ADS_IDEAL_HIGH, ADS_ZERO_HIGH,
    ).to_numpy()

    # Primary and robustness formulations.
    df["SPI_sensor_full"] = (
        0.40*df["A_affinity"] + 0.40*df["C_charge"] + 0.20*df["R_log_recovery"]
    )
    df["SPI_sensor_no_recovery"] = 0.50*df["A_affinity"] + 0.50*df["C_charge"]
    df["SPI_sensor_window"] = (
        0.40*df["A_window"] + 0.40*df["C_charge"] + 0.20*df["R_log_recovery"]
    )
    df["SPI_device_revised"] = (
        0.30*df["A_affinity"] + 0.30*df["C_charge"]
        + 0.20*df["R_log_recovery"] + 0.20*df["Electronic_accessibility"]
    )

    rank_columns = {
        "SPI_sensor_full": "Rank_full",
        "SPI_sensor_no_recovery": "Rank_no_recovery",
        "SPI_sensor_window": "Rank_window",
        "SPI_device_revised": "Rank_device_revised",
    }
    for score, rank in rank_columns.items():
        df[rank] = df[score].rank(ascending=False, method="min").astype("Int64")
    return df


def correlation(x, y, method):
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan, np.nan, len(z)
    fn = {"Pearson": pearsonr, "Spearman": spearmanr, "Kendall_tau_b": kendalltau}[method]
    coefficient, p = fn(z["x"], z["y"])
    return float(coefficient), float(p), len(z)


def top_k_overlap(rank_a, rank_b, k=TOP_K):
    a = set(rank_a.nsmallest(k).index)
    b = set(rank_b.nsmallest(k).index)
    return len(a & b), len(a & b) / k


def compare_rankings(df):
    comparisons = [
        ("Full vs recovery-excluded", "Rank_full", "Rank_no_recovery"),
        ("Full vs adsorption-window", "Rank_full", "Rank_window"),
        ("Full sensor vs revised device", "Rank_full", "Rank_device_revised"),
    ]
    rows = []
    for label, a, b in comparisons:
        rho, rho_p, n = correlation(df[a], df[b], "Spearman")
        tau, tau_p, _ = correlation(df[a], df[b], "Kendall_tau_b")
        overlap_n, overlap_fraction = top_k_overlap(df[a], df[b])
        rows.append({
            "Comparison": label, "Spearman_rho": rho, "Spearman_p": rho_p,
            "Kendall_tau_b": tau, "Kendall_p": tau_p,
            "Top5_overlap_n": overlap_n,
            "Top5_overlap_fraction": overlap_fraction, "N": n,
        })
    return pd.DataFrame(rows)


def redundancy_tests(df):
    rows = []
    for method in ["Pearson", "Spearman"]:
        r, p, n = correlation(df["Abs_Eads_eV"], df["log10_Recovery_s"], method)
        rows.append({
            "Variables": "|Eads| vs log10(recovery time)",
            "Method": method, "Coefficient": r, "p_value": p, "N": n,
        })
    z = df[["Abs_Eads_eV", "log10_Recovery_s"]].dropna()
    pair_r = z.corr().iloc[0, 1] if len(z) >= 3 else np.nan
    vif = np.inf if pd.isna(pair_r) or np.isclose(abs(pair_r), 1) else 1/(1-pair_r**2)
    rows.append({
        "Variables": "Two-variable collinearity", "Method": "VIF",
        "Coefficient": vif, "p_value": np.nan, "N": len(z),
    })
    return pd.DataFrame(rows)


def weight_sensitivity(df):
    score_table = df[["Candidate_Key", "System_ID", "Medium", "Family"]].copy()
    rank_table = score_table.copy()
    for name, (wa, wc, wr) in SCENARIOS.items():
        score = wa*df["A_affinity"] + wc*df["C_charge"] + wr*df["R_log_recovery"]
        score_table[name] = score
        rank_table[name] = score.rank(ascending=False, method="min")

    base = rank_table["Baseline_40_40_20"]
    metrics = []
    for name in SCENARIOS:
        rho, rho_p, n = correlation(base, rank_table[name], "Spearman")
        tau, tau_p, _ = correlation(base, rank_table[name], "Kendall_tau_b")
        overlap_n, overlap_fraction = top_k_overlap(base, rank_table[name])
        winner_idx = score_table[name].idxmax()
        metrics.append({
            "Scenario": name, "Winner": df.loc[winner_idx, "Candidate_Key"],
            "Spearman_rho": rho, "Spearman_p": rho_p,
            "Kendall_tau_b": tau, "Kendall_p": tau_p,
            "Top5_overlap_n": overlap_n,
            "Top5_overlap_fraction": overlap_fraction, "N": n,
        })
    return score_table, rank_table, pd.DataFrame(metrics)


def compare_with_old_ranks(df, path):
    if not path.exists():
        return pd.DataFrame({"Note": ["Old rank workbook was not found."]})
    old = pd.read_excel(path, sheet_name=0)
    if not {"System_ID", "Medium", "Rank_SPI1", "Rank_SPI2"}.issubset(old.columns):
        return pd.DataFrame({"Note": ["Expected old-rank columns were not found."]})
    old["Medium_clean"] = old["Medium"].astype(str).str.strip().str.lower()
    old["Candidate_Key"] = old["System_ID"].astype(str).str.strip() + " | " + old["Medium_clean"]
    keep = old[["Candidate_Key", "Rank_SPI1", "Rank_SPI2", "SPI_sensor", "SPI_device"]].copy()
    new = df[[
        "Candidate_Key", "SPI_sensor_full", "SPI_device_revised",
        "Rank_full", "Rank_device_revised",
    ]].copy()
    merged = new.merge(keep, on="Candidate_Key", how="left")
    merged["Sensor_rank_change_new_minus_old"] = merged["Rank_full"] - merged["Rank_SPI1"]
    merged["Device_rank_change_new_minus_old"] = merged["Rank_device_revised"] - merged["Rank_SPI2"]
    return merged.sort_values("Rank_full")


def monte_carlo_ranking(df):
    valid = df[[
        "Candidate_Key", "Abs_Eads_eV", "Abs_Charge_Transfer_e", "log10_Recovery_s"
    ]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(valid)
    ranks = np.empty((N_MONTE_CARLO, n), dtype=np.int16)

    for i in range(N_MONTE_CARLO):
        e = np.clip(
            valid["Abs_Eads_eV"].to_numpy()*rng.normal(1, EADS_RELATIVE_SD, n), 0, None
        )
        q = np.clip(
            valid["Abs_Charge_Transfer_e"].to_numpy()*rng.normal(1, CHARGE_RELATIVE_SD, n), 0, None
        )
        log_tau = valid["log10_Recovery_s"].to_numpy() + rng.normal(0, LOG10_TAU_ABSOLUTE_SD, n)
        A = (e-e.min())/(e.max()-e.min()) if e.max() != e.min() else np.full(n, 0.5)
        C = (q-q.min())/(q.max()-q.min()) if q.max() != q.min() else np.full(n, 0.5)
        R = 1-(log_tau-log_tau.min())/(log_tau.max()-log_tau.min()) if log_tau.max() != log_tau.min() else np.full(n, 0.5)
        score = 0.40*A + 0.40*C + 0.20*R
        order = np.argsort(-score, kind="stable")
        rank = np.empty(n, dtype=np.int16)
        rank[order] = np.arange(1, n+1)
        ranks[i] = rank

    summary = pd.DataFrame({
        "Candidate_Key": valid["Candidate_Key"],
        "P_Rank1": (ranks == 1).mean(axis=0),
        "P_Top5": (ranks <= min(TOP_K, n)).mean(axis=0),
        "Median_Rank": np.median(ranks, axis=0),
        "Rank_95pct_low": np.quantile(ranks, 0.025, axis=0),
        "Rank_95pct_high": np.quantile(ranks, 0.975, axis=0),
    }).sort_values(["P_Rank1", "P_Top5"], ascending=False)
    settings = pd.DataFrame([{
        "Iterations": N_MONTE_CARLO, "Eads_relative_SD": EADS_RELATIVE_SD,
        "Charge_relative_SD": CHARGE_RELATIVE_SD,
        "log10_tau_absolute_SD": LOG10_TAU_ABSOLUTE_SD,
        "Random_seed": RANDOM_STATE,
    }])
    return summary, settings


def build_models():
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.03, max_depth=2,
            loss="huber", random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not installed: RF and GB will still run.")
    return models


def ml_analysis(df):
    target = "SPI_sensor_full"
    # Constituent raw descriptors are intentionally included: this is a
    # reconstruction/surrogate task. The manuscript must disclose this leakage.
    candidate_features = [
        "Dopant_Count", "Total Atoms", "Distance_A", "HOMO_eV", "LUMO_eV",
        "Bandgap_eV", "Mulliken_Charge_Dopant", "Charge_Transfer_e",
        "E_ads_DFT_eV", "Recoverytime_sec", "electronegativity_eV",
        "chemical potential_eV", "hardness_eV", "electrophilicity index_eV",
    ]
    features = [c for c in candidate_features if c in df.columns]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[target]
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    groups = df.loc[mask, "Family"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    holdout_rows, cv_rows, fitted = [], [], {}

    for model_name, model in build_models().items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[model_name] = pipe
        holdout_rows.append({
            "Model": model_name, "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred)**0.5,
            "N_train": len(y_train), "N_test": len(y_test),
        })
        cv = cross_validate(
            pipe, X, y, cv=folds,
            scoring={"R2": "r2", "MAE": "neg_mean_absolute_error",
                     "RMSE": "neg_root_mean_squared_error"},
        )
        for fold in range(5):
            cv_rows.append({
                "Model": model_name, "Fold": fold+1,
                "R2": cv["test_R2"][fold],
                "MAE": -cv["test_MAE"][fold],
                "RMSE": -cv["test_RMSE"][fold],
            })

    holdout = pd.DataFrame(holdout_rows).sort_values("RMSE")
    cv_detail = pd.DataFrame(cv_rows)
    cv_summary = cv_detail.groupby("Model").agg(
        R2_mean=("R2", "mean"), R2_SD=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
    ).reset_index()

    # Leave-one-family-out evaluation with Gradient Boosting.
    family_rows = []
    for family in sorted(groups.dropna().unique()):
        train_mask = groups != family
        test_mask = groups == family
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=250, learning_rate=0.03, max_depth=2,
                loss="huber", random_state=RANDOM_STATE,
            )),
        ])
        pipe.fit(X.loc[train_mask], y.loc[train_mask])
        pred = pipe.predict(X.loc[test_mask])
        family_rows.append({
            "Held_out_family": family, "N_test": int(test_mask.sum()),
            "R2": r2_score(y.loc[test_mask], pred),
            "MAE": mean_absolute_error(y.loc[test_mask], pred),
            "RMSE": mean_squared_error(y.loc[test_mask], pred)**0.5,
        })

    # SHAP when installed; permutation importance otherwise.
    best_name = holdout.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    try:
        import shap
        X_imp = pd.DataFrame(
            best_pipe.named_steps["imputer"].transform(X), columns=features, index=X.index
        )
        explainer = shap.Explainer(best_pipe.named_steps["model"], X_imp)
        values = explainer(X_imp)
        importance = pd.DataFrame({
            "Feature": features,
            "Mean_absolute_SHAP": np.abs(values.values).mean(axis=0),
            "Importance_method": "SHAP",
        }).sort_values("Mean_absolute_SHAP", ascending=False)
        np.save(OUTPUT_DIR/"shap_values.npy", values.values)
    except Exception as exc:
        result = permutation_importance(
            best_pipe, X_test, y_test, n_repeats=30,
            random_state=RANDOM_STATE, scoring="neg_mean_absolute_error",
        )
        importance = pd.DataFrame({
            "Feature": features,
            "Permutation_importance": result.importances_mean,
            "Importance_SD": result.importances_std,
            "Importance_method": "Permutation (SHAP unavailable)",
        }).sort_values("Permutation_importance", ascending=False)
        (OUTPUT_DIR/"SHAP_NOT_RUN.txt").write_text(
            f"SHAP was not run: {exc}\nInstall it with: pip install shap\n",
            encoding="utf-8",
        )
    return holdout, cv_detail, cv_summary, pd.DataFrame(family_rows), importance


def exploratory_electronic_comparison(df):
    rows = []
    for method in ["Spearman", "Kendall_tau_b"]:
        r, p, n = correlation(
            df["SPI_sensor_full"], df["Electronic_accessibility"], method
        )
        rows.append({
            "Comparison": "SPI_sensor vs N(1/Eg)", "Method": method,
            "Coefficient": r, "p_value": p, "N": n,
            "Interpretation": (
                "Exploratory association only; 1/Eg is not adsorption-induced "
                "response and is not experimental validation."
            ),
        })
    return pd.DataFrame(rows)


def create_figures(df, scenario_metrics, mc, cv_detail, importance):
    sns.set_theme(style="whitegrid")
    top = df.nlargest(15, "SPI_sensor_full").sort_values("SPI_sensor_full")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Candidate_Key"], top["SPI_sensor_full"], color="#1769aa")
    ax.set_xlabel("Revised SPI_sensor")
    ax.set_ylabel("")
    ax.set_title("Top 15 candidates after log-recovery normalization")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"Figure_1_Top15_Revised_SPI.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    p = mc.head(15).sort_values("P_Top5")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(p["Candidate_Key"], p["P_Top5"], color="#2e8b57")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of a top-five rank")
    ax.set_ylabel("")
    ax.set_title("Monte Carlo rank robustness")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"Figure_2_Monte_Carlo_Top5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cv_plot = cv_detail.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=cv_plot, x="Model", y="R2", ax=ax, color="#90caf9")
    sns.stripplot(data=cv_plot, x="Model", y="R2", ax=ax, color="black", size=5)
    ax.set_title("Five-fold cross-validation of revised SPI surrogate models")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"Figure_3_FiveFold_CV.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    value_col = "Mean_absolute_SHAP" if "Mean_absolute_SHAP" in importance else "Permutation_importance"
    imp = importance.head(12).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["Feature"], imp[value_col], color="#ef6c00")
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_ylabel("")
    ax.set_title("Feature importance for SPI surrogate reconstruction")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"Figure_4_Feature_Importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Place {RAW_FILE.name} beside this script.")

    df = prepare_raw_data(RAW_FILE)
    ranking_comparison = compare_rankings(df)
    redundancy = redundancy_tests(df)
    scenario_scores, scenario_ranks, scenario_metrics = weight_sensitivity(df)
    old_new = compare_with_old_ranks(df, OLD_RANK_FILE)
    mc, mc_settings = monte_carlo_ranking(df)
    holdout, cv_detail, cv_summary, family_cv, importance = ml_analysis(df)
    electronic = exploratory_electronic_comparison(df)

    methodology = pd.DataFrame({
        "Item": [
            "Primary raw source", "Old-rank source", "Adsorption term",
            "Charge term", "Recovery term", "Primary sensor index",
            "Recovery-excluded index", "Device index", "ML interpretation",
            "Independent-validation limitation", "Recovery temperature",
            "Recovery attempt frequency",
        ],
        "Definition": [
            RAW_FILE.name, OLD_RANK_FILE.name,
            "A = MinMax(|Eads|)", "C = MinMax(|Q|)",
            "R = 1 - MinMax(log10(tau/s))",
            "SPI_sensor = 0.40A + 0.40C + 0.20R",
            "SPI_no_recovery = 0.50A + 0.50C",
            "SPI_device = 0.30A + 0.30C + 0.20R + 0.20MinMax(1/Eg)",
            "Surrogate reconstruction, not experimental sensor prediction",
            "The file lacks paired bare/adsorbed gaps; N(1/Eg) is exploratory only",
            f"{TEMPERATURE_K} K",
            f"{ATTEMPT_FREQUENCY_S:.1e} s^-1",
        ],
    })

    output_xlsx = OUTPUT_DIR/"Revised_SPI_Reviewer_Analysis.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
        df.sort_values("Rank_full").to_excel(writer, sheet_name="Revised_Ranking", index=False)
        old_new.to_excel(writer, sheet_name="Old_vs_New_Ranks", index=False)
        ranking_comparison.to_excel(writer, sheet_name="Index_Robustness", index=False)
        redundancy.to_excel(writer, sheet_name="Redundancy_Tests", index=False)
        scenario_scores.to_excel(writer, sheet_name="Scenario_Scores", index=False)
        scenario_ranks.to_excel(writer, sheet_name="Scenario_Ranks", index=False)
        scenario_metrics.to_excel(writer, sheet_name="Sensitivity_Metrics", index=False)
        mc.to_excel(writer, sheet_name="Monte_Carlo_Ranks", index=False)
        mc_settings.to_excel(writer, sheet_name="Monte_Carlo_Settings", index=False)
        holdout.to_excel(writer, sheet_name="ML_Holdout", index=False)
        cv_detail.to_excel(writer, sheet_name="ML_5Fold_Detail", index=False)
        cv_summary.to_excel(writer, sheet_name="ML_5Fold_Summary", index=False)
        family_cv.to_excel(writer, sheet_name="Leave_Family_Out", index=False)
        importance.to_excel(writer, sheet_name="SHAP_or_Importance", index=False)
        electronic.to_excel(writer, sheet_name="Electronic_Comparison", index=False)

    create_figures(df, scenario_metrics, mc, cv_detail, importance)
    winner = df.loc[df["SPI_sensor_full"].idxmax()]
    summary = {
        "systems_analyzed": int(len(df)),
        "revised_rank_1": winner["Candidate_Key"],
        "revised_rank_1_score": float(winner["SPI_sensor_full"]),
        "output_workbook": str(output_xlsx),
        "warning": "ML metrics are surrogate-reconstruction metrics, not experimental accuracy.",
    }
    (OUTPUT_DIR/"run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
