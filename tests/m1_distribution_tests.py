# core import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_test import BaseModelTest
# default imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# stat imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import statsmodels.api as sm
# module imports
from tqdm import tqdm
import itertools
from typing import List, Dict, Union
from tests.psi_calculator import PSICalculator, PSICalculatorExtended
from core.config_validation import validate_config, ConfigValidationError

def _frames_from_cfg(df: pd.DataFrame, cfg):
    sc = cfg.columns.sample_column
    # train/test определяем по значениям sample: 'train' / 'test' (в нижнем регистре)
    if sc is None or sc not in df.columns:
        raise ValueError("В конфиге задан sample_column, но его нет в df")
    s = df[sc].astype(str).str.lower()
    df_train = df[s.eq('train')]
    df_test  = df[s.eq('test')]
    df_genpop = df  # genpop — весь df
    return df_train, df_test, df_genpop

def _features_from_cfg(df: pd.DataFrame, cfg):
    num = cfg.columns.numeric_features or []
    cat = cfg.columns.categorical_features or []
    features = list(dict.fromkeys([*num, *cat]))  # порядок + удаление дублей
    return features, cat

# Реализация тестов M 1.1 и M 1.2: сравнение TRAIN и OOS против GENPOP
class M11_TrainVsGenpopPSITest(BaseModelTest):
    def __init__(self, psi_calc: PSICalculatorExtended):
        super().__init__("M 1.1", "TRAIN vs GENPOP PSI Stability")
        self.psi_calc = psi_calc

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, _, df_genpop = _frames_from_cfg(df, cfg)
        features, cat_features = _features_from_cfg(df, cfg)
        target_column = cfg.columns.target_column

        psi_results = {}
        all_psi_tables = {}
        for feature in tqdm(features, desc="M 1.1 - Processing Features"):
            psi_result = self.psi_calc.calculate(
                expected=df_train[feature],
                actual=df_genpop[feature],
                target_expected=df_train[target_column],
                target_actual=df_genpop[target_column],
                is_categorical=feature in cat_features
            )
            psi_results[feature] = self.psi_calc.generate_html_block(psi_result, feature)
            all_psi_tables[feature] = psi_result

        html, signal = self.psi_calc.generate_dashboard_table(all_psi_tables, return_signal=True)
        psi_results["DASHBOARD"] = html
        self.test_signal = signal
        return psi_results



class M12_OOSVsGenpopPSITest(BaseModelTest):
    def __init__(self, psi_calc: PSICalculatorExtended):
        super().__init__("M 1.2", "OOS vs GENPOP PSI Stability")
        self.psi_calc = psi_calc

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        _, df_oot, df_genpop = _frames_from_cfg(df, cfg)
        features, cat_features = _features_from_cfg(df, cfg)
        target_column = cfg.columns.target_column

        psi_results = {}
        all_psi_tables = {}
        for feature in tqdm(features, desc="M 1.2 - Processing Features"):
            psi_result = self.psi_calc.calculate(
                expected=df_oot[feature],
                actual=df_genpop[feature],
                target_expected=df_oot[target_column],
                target_actual=df_genpop[target_column],
                is_categorical=feature in cat_features
            )
            psi_results[feature] = self.psi_calc.generate_html_block(psi_result, feature)
            all_psi_tables[feature] = psi_result

        html, signal = self.psi_calc.generate_dashboard_table(all_psi_tables, return_signal=True)
        psi_results["DASHBOARD"] = html
        self.test_signal = signal
        return psi_results



# --- M 1.3 — Max Date Recency ---
class M13_MaxDateRecencyTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 1.3", "Max Date Recency Check")

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        date_column = cfg.columns.date_column

        max_date = pd.to_datetime(df[date_column]).max()
        today = pd.Timestamp.today().normalize()
        days_diff = (today - max_date).days

        if days_diff <= 30:
            signal = "green"
        elif days_diff <= 90:
            signal = "yellow"
        else:
            signal = "red"

        self.test_signal = signal

        html = f"""
        <h4>Max Date Recency</h4>
        <p><b>Max date in dataset:</b> {max_date.date()}</p>
        <p><b>Today:</b> {today.date()}</p>
        <p><b>Days difference:</b> <span style='color:red'><b>{days_diff} days</b></span></p>
        """
        return {"date_recency": html}


# --- M 1.4 — Sample Naming Control ---
class M14_DataQualityOverviewTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 1.4", "Data Quality Overview")

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        date_column = cfg.columns.date_column
        sample_column = cfg.columns.sample_column
        primary_key = cfg.columns.id_column
        target_column = cfg.columns.target_column

        if cfg.task == "classification":
            predict_column = cfg.columns.prediction_column
        else:
            predict_column = cfg.columns.score_column

        valid_samples = ['train', 'test']  # можно расширить конфигом при необходимости

        html_blocks = []
        signal_levels = []

        # --- Data Summary ---
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            summary_data = []
            for sample in df[sample_column].dropna().unique():
                sample_df = df[df[sample_column] == sample]
                summary_data.append([
                    sample,
                    len(sample_df),
                    sample_df.duplicated().sum(),
                    sample_df[date_column].min(),
                    sample_df[date_column].max()
                ])
            summary_df = pd.DataFrame(summary_data, columns=["Sample", "Observations", "Duplicates", "Min Date", "Max Date"])
            html_blocks.append("<h4>Data Summary</h4>" + summary_df.to_html(index=False))
        except Exception as e:
            html_blocks.append(f"<p style='color:red'>Error in Data Summary: {str(e)}</p>")
            signal_levels.append("red")

        # --- Sample Overlay ---
        try:
            if isinstance(primary_key, str):
                key_cols = [primary_key]
            else:
                key_cols = list(primary_key)

            tmp = df[key_cols].astype(str).agg("_".join, axis=1)
            df["__key__"] = tmp + "_" + df[sample_column].astype(str)
            overlaps = []
            for a, b in itertools.combinations(df[sample_column].dropna().unique(), 2):
                keys_a = df[df[sample_column] == a]["__key__"].unique()
                keys_b = df[df[sample_column] == b]["__key__"].unique()
                intersection = np.intersect1d(keys_a, keys_b)
                overlaps.append({"Samples": f"{a} vs {b}", "Intersections": len(intersection)})
            df.drop("__key__", axis=1, inplace=True)
            overlay_df = pd.DataFrame(overlaps)
            html_blocks.append("<h4>Sample Overlap</h4>" + overlay_df.to_html(index=False))
        except Exception as e:
            html_blocks.append(f"<p style='color:red'>Error in Sample Overlay: {str(e)}</p>")
            signal_levels.append("red")

        # --- Observed vs Predicted ---
        try:
            for col in [predict_column, target_column]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='raise')
            obs_pred = []
            for sample in df[sample_column].dropna().unique():
                group = df[df[sample_column] == sample]
                obs = group[target_column].sum()
                pred = group[predict_column].sum()
                dev = (pred - obs) / obs * 100 if obs != 0 else np.nan
                obs_pred.append({
                    "Sample": sample,
                    "Observed": obs,
                    "Prediction": pred,
                    "Deviation (%)": round(dev, 2) if pd.notnull(dev) else np.nan
                })
            obs_df = pd.DataFrame(obs_pred)
            html_blocks.append("<h4>Observed vs Predicted</h4>" + obs_df.to_html(index=False))
        except Exception as e:
            html_blocks.append(f"<p style='color:red'>Error in Obs vs Pred: {str(e)}</p>")
            signal_levels.append("red")

        # --- Sample Name Check ---
        try:
            actual_samples = df[sample_column].dropna().astype(str).str.lower().unique().tolist()
            expected_samples = [s.lower() for s in valid_samples]
            unknown = [s for s in actual_samples if s not in expected_samples]
            html_blocks.append(f"""
                <h4>Sample Name Check</h4>
                <p><b>Expected samples:</b> {expected_samples}</p>
                <p><b>Found samples:</b> {actual_samples}</p>
                <p><b>Unknown samples:</b> <span style='color:red'>{unknown or '—'}</span></p>
            """)
            if unknown:
                signal_levels.append("red")
        except Exception as e:
            html_blocks.append(f"<p style='color:red'>Error in Sample Name Check: {str(e)}</p>")
            signal_levels.append("red")

        # --- Final Signal ---
        if "red" in signal_levels:
            self.test_signal = "red"
        elif "yellow" in signal_levels:
            self.test_signal = "yellow"
        else:
            self.test_signal = "green"

        return {"overview": "<hr>".join(html_blocks)}





# --- M 1.5 — Missing Values Analysis ---
class M15_MissingValuesTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 1.5", "Missing Values Overview")

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        features, _ = _features_from_cfg(df, cfg)

        nulls = df[features].isnull().sum()
        null_pct = (nulls / len(df)).round(4)
        total_missing_pct = null_pct.sum() / max(len(features), 1)

        if total_missing_pct == 0:
            signal = "green"
        elif total_missing_pct < 0.1:
            signal = "yellow"
        else:
            signal = "red"

        self.test_signal = signal

        result = pd.DataFrame({
            "Feature": nulls.index,
            "Missing Count": nulls.values,
            "Missing %": null_pct.values
        })
        result = result[result["Missing Count"] > 0]

        html = "<h4>Missing Values</h4>"
        html += result.to_html(index=False) if not result.empty else "<p><b style='color:green'>No missing values </b></p>"
        return {"missing_values": html}




