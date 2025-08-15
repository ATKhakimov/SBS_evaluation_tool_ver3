# tests_m5_stability.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_test import BaseModelTest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from io import BytesIO
import base64
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict
from tests.psi_calculator import PSICalculatorExtended
from tests.psi_calculator import PSICalculator


class M51_ModelGiniStabilityTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 5.1", "Model Gini Stability")

    def run(self, score_train: pd.Series, target_train: pd.Series,
                  score_test: pd.Series, target_test: pd.Series) -> Dict[str, str]:
        auc_train = roc_auc_score(target_train, score_train)
        auc_test = roc_auc_score(target_test, score_test)
        gini_train = 2 * auc_train - 1
        gini_test = 2 * auc_test - 1
        delta = abs(gini_train - gini_test)

        # Signal logic
        if delta > 0.1:
            signal = "red"
        elif delta > 0.05:
            signal = "orange"
        else:
            signal = "green"

        self.test_signal = signal

        html = f"""
        <h4>Model Gini Stability</h4>
        <p><b>Gini Train:</b> {gini_train:.4f}</p>
        <p><b>Gini Test:</b> {gini_test:.4f}</p>
        <p><b>Delta:</b> {delta:.4f}</p>
        <p><b>Signal:</b> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        """
        return {"combined": html}


class M52_FactorGiniStabilityTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 5.2", "Factor Gini Stability")

    def compute_gini(self, feature: pd.Series, target: pd.Series) -> float:
        df = pd.DataFrame({"feature": feature, "target": target}).sort_values("feature")
        df["cum_target"] = df["target"].cumsum()
        total = df["target"].sum()
        if total == 0:
            return 0.0
        df["lorentz"] = df["cum_target"] / total
        df["population"] = np.arange(1, len(df)+1) / len(df)
        gini = 2 * np.trapz(df["lorentz"], df["population"]) - 1
        return gini

    def run(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                  target_column: str, features: List[str]) -> Dict[str, str]:
        records = []
        for f in tqdm(features, desc="M 5.2 - Processing Features"):
            gini_train = self.compute_gini(df_train[f], df_train[target_column]) * 100
            gini_test = self.compute_gini(df_test[f], df_test[target_column]) * 100
            delta = abs(gini_train - gini_test)
            records.append((f, gini_train, gini_test, delta))

        df_gini = pd.DataFrame(records, columns=["Feature", "Gini Train", "Gini Test", "Delta"])

        # Signal logic
        red = df_gini[df_gini["Delta"] > 10]
        orange = df_gini[(df_gini["Delta"] > 5) & (df_gini["Delta"] <= 10)]
        if len(red) > 0:
            signal = "red"
        elif len(orange) > len(df_gini) * 0.2:
            signal = "orange"
        else:
            signal = "green"

        self.test_signal = signal

        html = f"""
        <h4>Factor Gini Stability</h4>
        <p><b>Signal:</b> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        {df_gini.to_html(index=False, float_format="%.2f")}
        """
        return {"combined": html}


class M53_ScorePSITest(BaseModelTest):
    def __init__(self, psi_calc: PSICalculatorExtended):
        super().__init__("M 5.3", "Score PSI Stability")
        self.psi_calc = psi_calc

    def run(self, train_df, oot_df, score_column, target_column) -> Dict[str, str]:
        psi_result = self.psi_calc.calculate(
            expected=train_df[score_column],
            actual=oot_df[score_column],
            target_expected=train_df[target_column],
            target_actual=oot_df[target_column],
            is_categorical=False
        )
        signal = self.psi_calc.get_psi_signal(psi_result)
        self.test_signal = signal

        html = self.psi_calc.generate_html_block(psi_result, feature_name="score")
        return {"score": html, "DASHBOARD": html}


class M54_FactorPSITest(BaseModelTest):
    def __init__(self, psi_calc: PSICalculatorExtended):
        super().__init__("M 5.4", "Factors PSI Stability")
        self.psi_calc = psi_calc

    def run(self, train_df, oot_df, target_column, features, categorical_features) -> Dict[str, str]:
        psi_results = {}
        all_psi_tables = {}

        for feature in tqdm(features, desc="M 5.4 - Processing Features"):
            result = self.psi_calc.calculate(
                expected=train_df[feature],
                actual=oot_df[feature],
                target_expected=train_df[target_column],
                target_actual=oot_df[target_column],
                is_categorical=feature in categorical_features
            )
            psi_results[feature] = self.psi_calc.generate_html_block(result, feature)
            all_psi_tables[feature] = result

        signal = self.psi_calc.get_group_psi_signal(all_psi_tables)
        self.test_signal = signal

        dashboard_html = self.psi_calc.generate_dashboard_table(all_psi_tables)
        psi_results["DASHBOARD"] = dashboard_html
        return psi_results
