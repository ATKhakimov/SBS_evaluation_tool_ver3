# core import
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_test import BaseModelTest

# default imports
import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from io import BytesIO
import base64

# stat imports (оставлены без использования — на будущее)
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score            # type: ignore
from scipy.stats import norm                         # type: ignore
import statsmodels.api as sm                         # type: ignore

# module imports
import itertools
from typing import List, Dict, Union

# NEW: валидация конфига
from core.config_validation import validate_config, ConfigValidationError


class M01_DataSummaryTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 0.1", "Data Summary")

    # NEW: теперь принимаем df + config
    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        """
        Сводка по датасету с разбивкой по sample: Observations, Duplicates, Min/Max Date.
        Берёт имена колонок из config.columns.*
        """
        cfg = validate_config(config, df)
        date_column = cfg.columns.date_column
        sample_column = cfg.columns.sample_column

        if date_column not in df.columns or sample_column not in df.columns:
            raise KeyError("В данных отсутствуют date_column или sample_column из config")

        # Переводим даты в datetime (мягко)
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Если попали NaT — предупреждаем
        if df[date_column].isnull().any():
            warnings.warn("Некоторые значения в date_column не приводятся к datetime (NaT).")

        summary_dict = {}
        index = ['Observations', 'Duplicates', 'Min Date', 'Max Date']

        for sample in df[sample_column].dropna().unique():
            sample_df = df[df[sample_column] == sample]
            summary_dict[sample] = [
                len(sample_df),
                sample_df.duplicated().sum(),
                sample_df[date_column].min(),
                sample_df[date_column].max()
            ]

        summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=index)

        html = "<h4>Data Summary</h4>" + summary_df.to_html()
        return {"summary": html}


class M02_SampleOverlayTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 0.2", "Sample Overlay")

    # NEW: теперь принимаем df + config (primary_key берём из config)
    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        """
        Считает пересечения объектов между сэмплами по primary key.
        primary_key берётся из config.columns.id_column (или columns.id_columns если задан список).
        """
        cfg = validate_config(config, df)
        sample_column = cfg.columns.sample_column
        # поддержим и одиночный id_column, и потенциальный список id_columns (если добавишь в config)
        primary_key = config.get("columns", {}).get("id_columns", None) or cfg.columns.id_column

        # нормализуем в список
        if isinstance(primary_key, str):
            key_cols = [primary_key]
        elif isinstance(primary_key, list):
            key_cols = primary_key
        else:
            raise ValueError("В config.columns ожидается id_column (str) или id_columns (List[str])")

        missing = [c for c in key_cols + [sample_column] if c not in df.columns]
        if missing:
            raise KeyError(f"В данных отсутствуют колонки: {missing}")
        if len(key_cols) > 10:
            raise ValueError("Слишком много ключевых колонок в id_columns — максимум 10.")

        df = df.copy()
        # ВАЖНО: формируем ключ ТОЛЬКО по первичному ключу (без sample),
        # иначе пересечения всегда будут пустыми.
        df["__key__"] = df[key_cols].astype(str).agg("_".join, axis=1)

        distinct_samples = df[sample_column].dropna().unique()
        overlaps = []
        for a, b in itertools.combinations(distinct_samples, 2):
            keys_a = df.loc[df[sample_column] == a, "__key__"].unique()
            keys_b = df.loc[df[sample_column] == b, "__key__"].unique()
            intersection = np.intersect1d(keys_a, keys_b)
            overlaps.append({"Samples": f"{a} vs {b}", "Intersections": int(len(intersection))})

        df.drop("__key__", axis=1, inplace=True)

        overlap_df = pd.DataFrame(overlaps) if overlaps else pd.DataFrame(columns=["Samples", "Intersections"])
        html = "<h4>Sample Overlap</h4>" + overlap_df.to_html(index=False)
        return {"overlap": html}


class M03_ObsVsPredTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 0.3", "Observed vs Predicted")

    # NEW: теперь принимаем df + config; колонку предсказания берём по task
    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        """
        По каждому sample считает сумму фактов, сумму предсказаний и отклонение (%).
        Для classification: используется prediction_column (метки 0/1).
        Для regression: используется score_column (вещественное предсказание).
        """
        cfg = validate_config(config, df)
        sample_column = cfg.columns.sample_column
        target_column = cfg.columns.target_column
        if cfg.task == "classification":
            predict_column = cfg.columns.prediction_column
            if predict_column is None:
                raise ConfigValidationError(["Для classification требуется columns.prediction_column (метки класса)."])
        else:
            predict_column = cfg.columns.score_column
            if predict_column is None:
                raise ConfigValidationError(["Для regression требуется columns.score_column (предсказание)."])

        needed = [predict_column, target_column, sample_column]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"В данных отсутствуют колонки: {missing}")

        df = df.copy()
        # Приведём к числовому типу столбцы с фактом/предсказанием (где уместно)
        for column in [predict_column, target_column]:
            if not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    df[column] = pd.to_numeric(df[column], errors='raise')
                except ValueError:
                    raise ValueError(f"Столбец {column} не является числовым. Перевод в числовой тип невозможен.")

        result = []
        for sample in df[sample_column].dropna().unique():
            group = df[df[sample_column] == sample]
            obs = float(group[target_column].sum())
            pred = float(group[predict_column].sum())
            deviation = (pred - obs) / obs * 100 if obs != 0 else np.nan
            result.append({
                "Sample": sample,
                "Observed": obs,
                "Prediction": pred,
                "Deviation (%)": round(deviation, 2) if pd.notnull(deviation) else np.nan
            })

        result_df = pd.DataFrame(result)
        html = "<h4>Observed vs Predicted</h4>" + result_df.to_html(index=False)
        return {"obs_pred": html}
