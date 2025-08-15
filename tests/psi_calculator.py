import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union, Tuple
from io import BytesIO
import base64
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_test import BaseModelTest

class PSICalculator:
    def __init__(self, bins: int = 10, binning_strategy: str = "quantile"):
        self.bins = bins
        self.strategy = binning_strategy

    def _bin_continuous(self, expected: pd.Series, actual: pd.Series):
        if self.strategy == 'quantile':
            edges = np.unique(np.percentile(expected.dropna(), np.linspace(0, 100, self.bins + 1)))
        elif self.strategy == 'uniform':
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())
            edges = np.linspace(min_val, max_val, self.bins + 1)
        else:
            raise ValueError("Unsupported binning strategy.")
        expected_binned = pd.cut(expected, bins=edges, include_lowest=True)
        actual_binned = pd.cut(actual, bins=edges, include_lowest=True)
        return expected_binned, actual_binned

    def _calculate_psi_from_bins(self, expected_counts, actual_counts):
        psi_df = pd.DataFrame({
            'bin': expected_counts.index.astype(str),
            'expected_pct': expected_counts.values,
            'actual_pct': actual_counts.reindex(expected_counts.index, fill_value=0).values
        })
        psi_df['psi'] = (psi_df['actual_pct'] - psi_df['expected_pct']) * np.log(
            np.where(
                psi_df['expected_pct'] == 0,
                1,
                np.where(psi_df['actual_pct'] == 0, 1e-6, psi_df['actual_pct'] / psi_df['expected_pct'])
            )
        )
        return psi_df

    def calculate(self, expected, actual,
                  target_expected: Optional[pd.Series] = None,
                  target_actual: Optional[pd.Series] = None,
                  is_categorical: Optional[bool] = None) -> Dict[str, pd.DataFrame]:

        def one_group(e, a):
            if is_categorical or e.dtype == 'object':
                expected_dist = e.value_counts(normalize=True)
                actual_dist = a.value_counts(normalize=True)
                return self._calculate_psi_from_bins(expected_dist, actual_dist)
            else:
                expected_binned, actual_binned = self._bin_continuous(e, a)
                expected_dist = expected_binned.value_counts(normalize=True, sort=False)
                actual_dist = actual_binned.value_counts(normalize=True, sort=False)
                return self._calculate_psi_from_bins(expected_dist, actual_dist)

        result = {'all': one_group(expected, actual)}
        if target_expected is not None and target_actual is not None:
            result['target_0'] = one_group(expected[target_expected == 0], actual[target_actual == 0])
            result['target_1'] = one_group(expected[target_expected == 1], actual[target_actual == 1])
        return result

    def _plot_distribution(self, df: pd.DataFrame, title: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_width = 0.4
        x = np.arange(len(df))
        ax.bar(x - bar_width / 2, df['expected_pct'], width=bar_width, label='Expected')
        ax.bar(x + bar_width / 2, df['actual_pct'], width=bar_width, label='Actual')
        ax.set_xticks(x)
        ax.set_xticklabels(df['bin'], rotation=45)
        ax.set_ylabel('Proportion')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode()

    def _plot_psi(self, df: pd.DataFrame, title: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(df['bin'], df['psi'])
        ax.set_title(title)
        ax.set_ylabel('PSI')
        x = np.arange(len(df))
        ax.set_xticks(x)
        ax.set_xticklabels(df['bin'], rotation=45)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode()

    def generate_html_block(self, psi_result: Dict[str, pd.DataFrame], feature_name: str) -> str:
        blocks = []
        for group, df in psi_result.items():
            total_psi = df["psi"].sum()
            dist_img = self._plot_distribution(df, f"{feature_name} - {group}")
            psi_img = self._plot_psi(df, f"{feature_name} - {group}")
            blocks.append(f"""
                <div class="psi-group-block">
                    <h4>Group: {group} (Total PSI = {total_psi:.4f})</h4>
                    {df.to_html(index=False, float_format="%.4f")}
                    <h5>Distribution</h5>
                    <img src="data:image/png;base64,{dist_img}"><br>
                    <h5>PSI by Bin</h5>
                    <img src="data:image/png;base64,{psi_img}">
                </div>
                <hr>
            """)
        return "\n".join(blocks)

class PSICalculatorExtended(PSICalculator):
    def get_group_psi_signal(self, psi_tables: Dict[str, Dict]) -> str:
        """
        Определяет итоговый сигнал по набору PSI-таблиц (по множеству фичей).

        Параметры:
            * psi_tables (Dict[str, Dict]): словарь, где ключ — имя фичи, значение — результат PSI (должен содержать 'total_psi').

        Возвращает:
            * str: Один из "green", "orange", "red"
        """
        psi_values = [res["total_psi"] for res in psi_tables.values() if "total_psi" in res]

        if not psi_values:
            return "red"  # если вообще ничего не посчитали

        red_count = sum(psi > 0.25 for psi in psi_values)
        orange_count = sum(0.1 < psi <= 0.25 for psi in psi_values)

        total = len(psi_values)
        if red_count > 0:
            return "red"
        elif orange_count / total > 0.2:
            return "orange"
        else:
            return "green"
   
    def get_psi_signal(self, psi_result: Dict) -> str:
        """
        Определяет итоговый сигнал (цвет светофора) по значению total_psi.
        
        Параметры:
            * psi_result (Dict): Словарь с результатами PSI, должен содержать ключ 'total_psi'

        Возвращает:
            * str: Один из 'green', 'orange', 'red'
        """
        psi_value = psi_result.get("total_psi", None)

        if psi_value is None:
            return "red"  # если по какой-то причине не посчитали PSI — это плохо

        if psi_value < 0.1:
            return "green"
        elif psi_value < 0.25:
            return "orange"
        else:
            return "red"
        
    def get_psi_color(self, psi_val: float) -> str:
        if psi_val < 0.1:
            return 'green'
        elif psi_val < 0.2:
            return 'yellow'
        else:
            return 'red'

    def generate_dashboard_table(self, psi_results_dict: Dict[str, Dict[str, pd.DataFrame]], return_signal: bool = False) -> Union[str, Tuple[str, str]]:
        def get_psi_color(psi_val):
            if psi_val < 0.1:
                return 'green'
            elif psi_val < 0.2:
                return 'yellow'
            else:
                return 'red'

        summary_rows = []
        color_buckets = {'green': [], 'yellow': [], 'red': []}
        for feature, group_dict in psi_results_dict.items():
            total_psi = group_dict["all"]["psi"].sum()
            color = get_psi_color(total_psi)
            color_buckets[color].append(feature)
            summary_rows.append(f"<tr><td>{feature}</td><td>{total_psi:.4f}</td><td style='color:{color}; font-weight:bold'>{color.upper()}</td></tr>")

        dashboard_html = f"""
        <h2>PSI Overview</h2>
        <table border="1" cellspacing="0" cellpadding="5">
            <tr><th>Feature</th><th>Total PSI</th><th>Status</th></tr>
            {''.join(summary_rows)}
        </table>
        <br>
        <h4>Green zone:</h4><p>{', '.join(color_buckets['green'])}</p>
        <h4>Yellow zone:</h4><p>{', '.join(color_buckets['yellow'])}</p>
        <h4>Red zone:</h4><p>{', '.join(color_buckets['red'])}</p>
        """

        if return_signal:
            if len(color_buckets['red']) > 0:
                return dashboard_html, 'red'
            elif len(color_buckets['yellow']) / len(psi_results_dict) > 0.2:
                return dashboard_html, 'yellow'
            else:
                return dashboard_html, 'green'
        else:
            return dashboard_html
    
    def generate_m5_3_and_m5_4_tests(
        self,
        train_df: pd.DataFrame,
        oot_df: pd.DataFrame,
        score_column: str,
        target_column: str,
        features: List[str],
        categorical_features: List[str]
    ) -> Dict[str, Dict[str, str]]:
        result = {
            "M 5.3": {},
            "M 5.4": {}
        }

        # M 5.3 — Score PSI Stability
        score_result = self.calculate(
            expected=train_df[score_column],
            actual=oot_df[score_column],
            target_expected=train_df[target_column],
            target_actual=oot_df[target_column],
            is_categorical=False
        )
        result["M 5.3"]["score"] = self.generate_html_block(score_result, feature_name="score")

        # M 5.4 — Factors PSI Stability
        psi_results_by_feature = {}
        for feature in features:
            feature_result = self.calculate(
                expected=train_df[feature],
                actual=oot_df[feature],
                target_expected=train_df[target_column],
                target_actual=oot_df[target_column],
                is_categorical=feature in categorical_features
            )
            psi_results_by_feature[feature] = feature_result
            result["M 5.4"][feature] = self.generate_html_block(feature_result, feature)

        result["M 5.4"]["DASHBOARD"] = self.generate_dashboard_table(psi_results_by_feature)
        return result
