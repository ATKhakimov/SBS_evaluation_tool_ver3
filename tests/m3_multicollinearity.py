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
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from typing import Optional, List, Dict

# NEW
from core.config_validation import validate_config, ConfigValidationError

# === helpers for M3 ===
def _frames_from_cfg(df: pd.DataFrame, cfg):
    sc = cfg.columns.sample_column
    if sc is None or sc not in df.columns:
        raise ValueError("В конфиге указан sample_column, но его нет в df")
    s = df[sc].astype(str).str.lower()
    df_train = df[s.eq('train')]
    df_test  = df[s.eq('test')]
    return df_train, df_test

def _numeric_features_from_cfg(cfg):
    feats = cfg.columns.numeric_features or []
    # для VIF нужны ТОЛЬКО численные колонки
    if not feats:
        raise ConfigValidationError(["Для M3 (VIF) требуется непустой список columns.numeric_features"])
    # уберём дубликаты, сохранив порядок
    return list(dict.fromkeys(feats))


class M31_FeatureTransformationTest(BaseModelTest): #NOT FOR USE!!!!!!!
    def __init__(self):
        super().__init__("M 3.1", "Feature Transformation Quality")

    def compute_feature_weights(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Вычисляет веса признаков с помощью логистической регрессии.
        Возвращает DataFrame с признаками, весами и нормированными весами.
        """
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="liblinear")
        model.fit(X, y)
        coef = model.coef_[0]
        feature_names = X.columns
        weights_df = pd.DataFrame({
            "Feature": feature_names,
            "Weight": coef,
            "AbsWeight": np.abs(coef)
        })
        weights_df["Normalized"] = weights_df["AbsWeight"] / weights_df["AbsWeight"].sum()
        return weights_df.sort_values("AbsWeight", ascending=False)

    def assign_flag(self, norm_weight: float) -> str:
        """
        Присваивает флаг признаку в зависимости от важности (нормированного веса).
        - > 20% -> green
        - 5%-20% -> orange
        - < 5% -> red
        """
        if norm_weight >= 0.2:
            return "green"
        elif norm_weight >= 0.05:
            return "orange"
        else:
            return "red"

    def compute_signal(self, flags: List[str]) -> str:
        """
        Генерирует итоговый сигнал по флагам.
        - Если есть красные -> red
        - Иначе если более 30% оранжевых -> orange
        - Иначе -> green
        """
        if "red" in flags:
            return "red"
        if flags.count("orange") / len(flags) > 0.3:
            return "orange"
        return "green"

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, str]:
        weights_df = self.compute_feature_weights(X, y)
        weights_df["Flag"] = weights_df["Normalized"].apply(self.assign_flag)

        signal = self.compute_signal(weights_df["Flag"].tolist())
        self.test_signal = signal

        summary_table = weights_df[["Feature", "Weight", "Normalized", "Flag"]].to_html(index=False, float_format="%.4f")

        html = f"""
        <h4>Feature Transformation Analysis</h4>
        <p><b>Signal:</b> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        {summary_table}
        """
        return {"combined": html}

class M33_MulticollinearityTest(BaseModelTest):
    def __init__(self, test_num='M 3.3', test_name='Multicollinearity Test'):
        super().__init__(test_num, test_name)

    def compute_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет коэффициенты VIF для каждого признака с прогрессбаром tqdm.

        Параметры:
            * df (pd.DataFrame): DataFrame с признаками.

        Возвращает:
            * pd.DataFrame: Таблица VIF.
        """
        X = sm.add_constant(df)
        vif_data = []

        for i in tqdm(range(1, X.shape[1]), desc="Calculating VIF"):  # пропускаем константу
            feature = X.columns[i]
            vif = variance_inflation_factor(X.values, i)
            vif_data.append((feature, vif))

        return pd.DataFrame(vif_data, columns=["Feature", "VIF"])

    def assign_flag(self, vif: float) -> str:
        if vif > 10:
            return "red"
        elif vif > 5:
            return "orange"
        else:
            return "green"

    def compute_signal(self, flags: List[str]) -> str:
        if "red" in flags:
            return "red"
        orange_count = flags.count("orange")
        total = len(flags)
        if orange_count / total > 0.2:
            return "orange"
        return "green"

    def plot_vif_comparison(self, merged: pd.DataFrame) -> str:
        x = np.arange(len(merged))
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.4
        ax.bar(x - bar_width / 2, merged["VIF_train"], width=bar_width, label="Train")
        ax.bar(x + bar_width / 2, merged["VIF_test"], width=bar_width, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["Feature"], rotation=45, ha="right")
        ax.set_ylabel("VIF")
        ax.set_title("VIF Comparison (Train vs Test)")
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        # 1) валидируем конфиг и режем выборки
        cfg = validate_config(config, df)
        df_train, df_test = _frames_from_cfg(df, cfg)
        features = _numeric_features_from_cfg(cfg)

        # 2) формируем X_train / X_test только по численным фичам
        missing_tr = [c for c in features if c not in df_train.columns]
        missing_te = [c for c in features if c not in df_test.columns]
        if missing_tr or missing_te:
            raise ValueError(f"Отсутствуют фичи в train/test: train_missing={missing_tr}, test_missing={missing_te}")

        X_train = df_train[features].copy()
        X_test  = df_test[features].copy()

        # 3) приводим к numeric и проверяем NaN (VIF не терпит пропусков)
        for X, name in [(X_train, "train"), (X_test, "test")]:
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        X[col] = pd.to_numeric(X[col], errors='raise')
                    except Exception:
                        raise ValueError(f"Колонка {col} в {name} не числовая и не конвертируется в numeric.")
            if X.isnull().any().any():
                bad = X.columns[X.isnull().any()].tolist()
                raise ValueError(f"В {name} обнаружены пропуски в колонках {bad}. "
                                f"Очистите/импутируйте данные для расчёта VIF.")

        # 4) считаем VIF (с обработкой ∞/NaN)
        vif_train = self.compute_vif(X_train)
        vif_test  = self.compute_vif(X_test) if X_test is not None and len(X_test) > 0 else None

        if vif_test is not None:
            merged = pd.merge(vif_train, vif_test, on="Feature", how="inner", suffixes=("_train", "_test"))
            # если какие-то фичи отвалились (редко), аккуратно добьём outer-merge и NaN → inf
            if merged.empty:
                merged = pd.merge(vif_train, vif_test, on="Feature", how="outer", suffixes=("_train", "_test"))
            # нормализуем бесконечности/NaN
            merged["VIF_train"] = merged["VIF_train"].replace([np.inf, -np.inf], np.nan)
            merged["VIF_test"]  = merged["VIF_test"].replace([np.inf, -np.inf], np.nan)
            merged["VIF_train"] = merged["VIF_train"].fillna(1e6)
            merged["VIF_test"]  = merged["VIF_test"].fillna(1e6)
            merged["Delta"] = (merged["VIF_train"] - merged["VIF_test"]).abs()
            merged["Flag_Train"] = merged["VIF_train"].apply(self.assign_flag)
            merged["Flag_Test"]  = merged["VIF_test"].apply(self.assign_flag)

            # флаг-сводка
            def count_flags(flag_series):
                return {"green": (flag_series == "green").sum(),
                        "orange": (flag_series == "orange").sum(),
                        "red": (flag_series == "red").sum()}
            train_flags = count_flags(merged["Flag_Train"])
            test_flags  = count_flags(merged["Flag_Test"])
            signal_train = self.compute_signal(merged["Flag_Train"].tolist())
            signal_test  = self.compute_signal(merged["Flag_Test"].tolist())
            self.test_signal = "red" if "red" in [signal_train, signal_test] else \
                            ("orange" if "orange" in [signal_train, signal_test] else "green")

            # листы фич по зонам (по train, как и раньше)
            def features_by_flag(flag_series, flag):
                return merged[flag_series == flag]["Feature"].tolist()
            features_green  = features_by_flag(merged["Flag_Train"], "green")
            features_orange = features_by_flag(merged["Flag_Train"], "orange")
            features_red    = features_by_flag(merged["Flag_Train"], "red")

            # HTML
            features_by_zone_html = f"""
            <h4>Feature Flags (Train)</h4>
            <p><b style='color:green'>Green Zone Features:</b> {', '.join(features_green) if features_green else 'None'}</p>
            <p><b style='color:orange'>Orange Zone Features:</b> {', '.join(features_orange) if features_orange else 'None'}</p>
            <p><b style='color:red'>Red Zone Features:</b> {', '.join(features_red) if features_red else 'None'}</p>
            """

            summary_table = f"""
            <h4>VIF Zones Summary</h4>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><th>Dataset</th><th style='color:green'>Green</th><th style='color:orange'>Orange</th><th style='color:red'>Red</th></tr>
                <tr><td>Train</td><td style='color:green'><b>{train_flags["green"]}</b></td>
                <td style='color:orange'><b>{train_flags["orange"]}</b></td>
                <td style='color:red'><b>{train_flags["red"]}</b></td></tr>
                <tr><td>Test</td><td style='color:green'><b>{test_flags["green"]}</b></td>
                <td style='color:orange'><b>{test_flags["orange"]}</b></td>
                <td style='color:red'><b>{test_flags["red"]}</b></td></tr>
            </table>
            """

            signal_html = f"""
            <p><b>Final Signal (Train):</b> <span style='color:{signal_train}; font-weight:bold'>{signal_train.upper()}</span></p>
            <p><b>Final Signal (Test):</b> <span style='color:{signal_test}; font-weight:bold'>{signal_test.upper()}</span></p>
            """

            vif_plot = self.plot_vif_comparison(merged)
            table_html = merged[["Feature", "VIF_train", "VIF_test", "Delta"]].to_html(index=False, float_format="%.2f")

            html = f"""
            <h4>Multicollinearity Test</h4>
            {signal_html}
            {summary_table}
            <br>
            {features_by_zone_html}
            <br>
            <h5>VIF Comparison Chart</h5>
            <img src="data:image/png;base64,{vif_plot}">
            <h5>VIF Table</h5>
            {table_html}
            """
            return {"combined": html}

        else:
            # кейс, когда тест-сэмпла нет → считаем только train и даём аккуратный вывод
            vif_train = vif_train.replace([np.inf, -np.inf], np.nan).fillna(1e6)
            vif_train["Flag_Train"] = vif_train["VIF"].apply(self.assign_flag)
            signal_train = self.compute_signal(vif_train["Flag_Train"].tolist())
            self.test_signal = signal_train

            table_html = vif_train.rename(columns={"VIF": "VIF_train"}).to_html(index=False, float_format="%.2f")
            signal_html = f"""
            <p><b>Final Signal (Train):</b> <span style='color:{signal_train}; font-weight:bold'>{signal_train.upper()}</span></p>
            <p><i>Test выборка не обнаружена (sample='test'). Показаны только метрики Train.</i></p>
            """

            html = f"""
            <h4>Multicollinearity Test (Train only)</h4>
            {signal_html}
            <h5>VIF Table</h5>
            {table_html}
            """
            return {"combined": html}
