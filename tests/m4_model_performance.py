# tests/m4_model_performance.py
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score, mean_squared_error
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from tqdm import tqdm
from typing import Dict, List, Callable, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта SHAP (может отсутствовать)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from core.base_test import BaseModelTest

# NEW — используем единый валидатор конфига
from core.config_validation import validate_config, ConfigValidationError  # <-- ваш модуль валидации

# === Общие хелперы для M4 ===
def _split_frames_from_cfg(df: pd.DataFrame, cfg):
    sc = cfg.columns.sample_column
    if sc is None or sc not in df.columns:
        raise ValueError("В конфиге задан sample_column, но его нет в df")
    s = df[sc].astype(str).str.lower()
    df_train = df[s.eq('train')]
    df_test  = df[s.eq('test')]
    df_gen   = df
    return df_train, df_test, df_gen

def _features_numeric(cfg):
    feats = cfg.columns.numeric_features or []
    return list(dict.fromkeys(feats))  # порядок+удаление дублей

def _get_y_cols(df: pd.DataFrame, cfg, *, for_classification=True):
    """
    Возвращает кортеж:
      - y_true (Series)
      - y_pred_labels (Series) — для классификации
      - y_score (Series) — prob для классификации или regression score для регрессии
    """
    y_true = df[cfg.columns.target_column]
    score_col = cfg.columns.score_column
    pred_col  = cfg.columns.prediction_column

    if for_classification:
        if pred_col and pred_col in df.columns:
            y_pred_labels = df[pred_col]
        elif score_col and score_col in df.columns:
            # fallback: порог 0.5 → метка
            y_pred_labels = (df[score_col] >= 0.5).astype(int)
        else:
            raise ConfigValidationError(["Для classification требуется prediction_column (метки) или score_column (prob)."])
        if not score_col or score_col not in df.columns:
            y_score = None
        else:
            y_score = df[score_col]
        return y_true, y_pred_labels, y_score
    else:
        # regression
        if not score_col or score_col not in df.columns:
            raise ConfigValidationError(["Для regression требуется columns.score_column (предсказание)."])
        return y_true, None, df[score_col]


# --- Config helpers (fixed for nested config["columns"]) ---
from typing import Sequence

def _cfg_section(cfg: dict) -> dict:
    # поддерживает как вложенный конфиг {"columns": {...}}, так и "плоский"
    if isinstance(cfg, dict) and "columns" in cfg and isinstance(cfg["columns"], dict):
        return cfg["columns"]
    return cfg

def _cfg_col(cfg: dict, key: str, default=None):
    sec = _cfg_section(cfg)
    val = sec.get(key, default) if isinstance(sec, dict) else default
    if val is None:
        raise ValueError(f"Config['{key}'] is required for this test.")
    return val

def _cfg_get_features(cfg: dict) -> list[str]:
    sec = _cfg_section(cfg)
    # приоритет: если где-то заведёшь features_all — используем его
    feats_all = sec.get("features_all")
    if feats_all:
        return list(feats_all)
    num = sec.get("numeric_features", []) or []
    cat = sec.get("categorical_features", []) or []
    return list(num) + list(cat)

def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in df: {missing}")

def _select_X(df: pd.DataFrame, cfg: dict, features: list[str] | None = None) -> pd.DataFrame:
    feats = features if features is not None else _cfg_get_features(cfg)
    _ensure_columns(df, feats)
    return df[feats]

def _subset_by_sample(df: pd.DataFrame, cfg: dict, sample_value: str) -> pd.DataFrame:
    sample_col = _cfg_col(cfg, "sample_column")
    return df[df[sample_col].astype(str).str.lower() == str(sample_value).lower()]

def _numeric_only(X: pd.DataFrame) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("No numeric features found; please specify numeric features in config.columns.numeric_features.")
    return numeric


class M41_MetricConfidenceIntervalTest(BaseModelTest):
    """
    Section 4.1. Algorithm Performance Quality
    Test 4.1.1. Confidence interval of quality metric on validation
    """
    def __init__(self, model_type: str = 'classification', n_iterations: int = 300):
        """
        Args:
            model_type (str): Model type: 'classification' or 'regression'.
            n_iterations (int): Number of bootstrap iterations.
        """
        super().__init__("M 4.1", "Metric Confidence Interval")
        if model_type not in ['classification', 'regression']:
            raise ValueError("model_type must be 'classification' or 'regression'")
        self.model_type = model_type
        self.n_iterations = n_iterations # May take long time

    def _bootstrap_metric(self, y_true: pd.Series, y_pred: pd.Series, metric_func: Callable) -> List[float]:
        """Calculate metric using bootstrap."""
        metric_values = []
        for _ in tqdm(range(self.n_iterations), desc="Bootstrapping metric"):
            y_true_sample, y_pred_sample = resample(y_true, y_pred)
            metric_value = metric_func(y_true_sample, y_pred_sample)
            metric_values.append(metric_value)
        return metric_values

    def _generate_plot(self, y_true: pd.Series, y_pred: pd.Series) -> str:
        """Generate scatter plot for regression."""
        plt.figure(figsize=(12, 8))  # Стандартный размер
        sns.scatterplot(x=y_true, y=y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs True Values")
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _generate_distribution_plot(self, bootstrap_values: List[float], original_metric: float) -> str:
        """Generate distribution plot of metric."""
        plt.figure(figsize=(12, 8))  # Стандартный размер
        sns.histplot(bootstrap_values, kde=True)
        plt.axvline(x=original_metric, color='red', linestyle='--', label=f'Original Metric: {original_metric:.3f}')
        plt.title("Bootstrap Distribution of Metric")
        plt.xlabel("Metric Value")
        plt.ylabel("Frequency")
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def run(self, df: pd.DataFrame, config: dict, sample: str = 'test') -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, df_gen = _split_frames_from_cfg(df, cfg)
        df_used = {"train": df_train, "test": df_test, "genpop": df_gen}.get(str(sample).lower(), df_test)

        if self.model_type == 'classification':
            y_true, y_pred_labels, _ = _get_y_cols(df_used, cfg, for_classification=True)
            key_metric_func = f1_score
            key_metric_name = "F1 Score"
            original_f1 = f1_score(y_true, y_pred_labels)
            original_precision = precision_score(y_true, y_pred_labels)
            original_recall = recall_score(y_true, y_pred_labels)
            html_metrics = pd.DataFrame({
                "Metric": ["F1 Score (ключевая)", "Precision", "Recall"],
                "Value": [original_f1, original_precision, original_recall]
            }).to_html(index=False, float_format="%.3f")
            y_true_boot = y_true
            y_pred_boot = y_pred_labels
        else:
            y_true, _, y_score = _get_y_cols(df_used, cfg, for_classification=False)
            key_metric_func = r2_score
            key_metric_name = "R2 Score"
            original_r2 = r2_score(y_true, y_score)
            original_mse = mean_squared_error(y_true, y_score)
            html_metrics = pd.DataFrame({
                "Metric": ["R2 Score (ключевая)", "MSE"],
                "Value": [original_r2, original_mse]
            }).to_html(index=False, float_format="%.3f")
            y_true_boot = y_true
            y_pred_boot = y_score

        # Bootstrap (предупреждение: долго)
        bootstrap_values = self._bootstrap_metric(y_true_boot, y_pred_boot, key_metric_func)
        mean_bootstrap = np.mean(bootstrap_values)
        left_interval = np.percentile(bootstrap_values, 5)
        right_interval = np.percentile(bootstrap_values, 95)
        original_metric = key_metric_func(y_true_boot, y_pred_boot)

        # Бейзлайн без внешнего аргумента: majority baseline (clf) или среднее (reg)
        if self.model_type == 'classification':
            majority = int(round(y_true.mean()))
            dummy_metric = key_metric_func(y_true_boot, pd.Series(majority, index=y_true_boot.index))
        else:
            dummy_metric = key_metric_func(y_true_boot, pd.Series(y_true_boot.mean(), index=y_true_boot.index))

        if left_interval <= original_metric <= right_interval:
            signal = "green"
        elif original_metric > dummy_metric:
            signal = "orange"
        else:
            signal = "red"
        self.test_signal = signal

        html_content = f"""
        <h4>Metric Confidence Interval</h4>
        <p><strong>Signal:</strong> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        <h5>{'Classification' if self.model_type=='classification' else 'Regression'} Metrics</h5>
        {html_metrics}
        """

        dist_plot_img = self._generate_distribution_plot(bootstrap_values, original_metric)
        html_content += f'<h5>Bootstrap results for {key_metric_name}</h5>' \
                        f"<table><tr><td>Original</td><td>{original_metric:.3f}</td></tr>" \
                        f"<tr><td>Mean bootstrap</td><td>{mean_bootstrap:.3f}</td></tr>" \
                        f"<tr><td>CI 5-95%</td><td>[{left_interval:.3f}; {right_interval:.3f}]</td></tr>" \
                        f"<tr><td>Dummy</td><td>{dummy_metric:.3f}</td></tr></table>" \
                        f'<h5>Bootstrap Distribution Plot</h5><img src="data:image/png;base64,{dist_plot_img}" />'
        return {"report": html_content}


class M42_SampleSizeAdequacyTest(BaseModelTest):
    """
    Тест 4.1.2. Проверка на недостаточность количества наблюдений
    """
    def __init__(self, model_type: str = 'classification', step: float = 0.1):
        super().__init__("M 4.2", "Sample Size Adequacy")
        self.model_type = model_type
        self.step = step  # Шаг увеличения доли наблюдений

    def run(self, df: pd.DataFrame, config: dict, model=None, model_class=None, sample: str = 'train') -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, df_gen = _split_frames_from_cfg(df, cfg)
        df_used = {"train": df_train, "test": df_test, "genpop": df_gen}.get(str(sample).lower(), df_train)

        feats = _features_numeric(cfg)
        if not feats:
            raise ConfigValidationError(["Для M 4.2 требуется непустой список columns.numeric_features"])
        X = df_used[feats].copy()
        y = df_used[cfg.columns.target_column].copy()

        # Приводим к numeric и убираем NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        if X.isnull().any().any():
            X = X.fillna(X.median(numeric_only=True))
        if y.isnull().any():
            y = y.fillna(y.median() if self.model_type=='regression' else y.mode().iloc[0])

        if model is None:
            if model_class is not None:
                model = model_class()
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42) if self.model_type=='classification' \
                        else RandomForestRegressor(n_estimators=200, random_state=42)
        # ВНИМАНИЕ: Может работать долго из-за множественного переобучения модели
        sample_sizes = np.arange(self.step, 1.01, self.step)
        scores_mean = []
        scores_std = []
        
        for size in tqdm(sample_sizes, desc="Testing different sample sizes"):
            n_samples = int(len(X) * size)
            if n_samples < 10:  # Минимальный размер выборки
                continue
                
            # Случайное разбиение на обучение и тест
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_train = X.iloc[indices]
            y_train = y.iloc[indices]
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            if self.model_type == 'classification':
                score = f1_score(y_train, model.predict(X_train))
            else:
                score = r2_score(y_train, model.predict(X_train))
                
            scores_mean.append(score)
            scores_std.append(0)  # Можно добавить кросс-валидацию для расчета std
        
        # Анализ тренда для определения сигнала
        if len(scores_mean) >= 3:
            # Проверяем, есть ли плато в последних 30% точек
            plateau_start = int(len(scores_mean) * 0.7)
            plateau_scores = scores_mean[plateau_start:]
            
            if plateau_scores:
                score_variation = np.std(plateau_scores) / np.mean(plateau_scores) if np.mean(plateau_scores) > 0 else 1
                improvement_trend = (scores_mean[-1] - scores_mean[0]) / abs(scores_mean[0]) if scores_mean[0] != 0 else 0
                
                if score_variation < 0.05 and improvement_trend < 0.1:  # Стабильное плато
                    self.test_signal = "green"
                elif score_variation < 0.1 or improvement_trend < 0.2:  # Умеренное улучшение
                    self.test_signal = "orange" 
                else:  # Продолжающийся рост
                    self.test_signal = "red"
            else:
                self.test_signal = "orange"
        else:
            self.test_signal = "orange"
        
        # Создание графика
        plt.figure(figsize=(12, 8))  # Стандартный размер
        plt.plot(sample_sizes[:len(scores_mean)], scores_mean, 'b-', label='Metric Score')
        plt.fill_between(sample_sizes[:len(scores_mean)], 
                        np.array(scores_mean) - np.array(scores_std),
                        np.array(scores_mean) + np.array(scores_std), 
                        alpha=0.3)
        plt.xlabel('Training Sample Size Fraction')
        plt.ylabel('Model Score')
        plt.title('Learning Curve: Score vs Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>', "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>', "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]
        
        html_content = f"""
        <h4>Sample Size Adequacy Test {signal_color}</h4>
        <p>Analysis of model quality dependence on training sample size.</p>
        <img src="data:image/png;base64,{plot_img}" />
        <p><strong>Interpretation:</strong> If the graph plateaus when approaching 100% of data, 
        the sample size is sufficient. A positive trend indicates potential improvement with more data.</p>
        <p><strong>Signal:</strong> {signal_color} - {'Sample size is adequate' if self.test_signal == 'green' else 'Moderate improvement possible' if self.test_signal == 'orange' else 'More data needed'}</p>
        """
        
        return {"report": html_content}


class M43_CalibrationTest(BaseModelTest):
    """
    Тест 4.1.3. Соответствие наблюдаемого и предсказанного таргета (калибровка модели)
    --- Только для бинарной классификации ---
    """
    def __init__(self, n_bins: int = 10):
        super().__init__("M 4.3", "Model Calibration")
        self.n_bins = n_bins

    def run(self, df: pd.DataFrame, config: dict, sample: str = 'test') -> Dict[str, str]:
        cfg = validate_config(config, df)
        if cfg.task != 'classification':
            self.test_signal = "red"
            return {"report": "<h4>Model Calibration Test</h4><p>Applicable only for binary classification.</p>"}

        df_train, df_test, df_gen = _split_frames_from_cfg(df, cfg)
        df_used = {"train": df_train, "test": df_test, "genpop": df_gen}.get(str(sample).lower(), df_test)

        y_true, _, y_prob = _get_y_cols(df_used, cfg, for_classification=True)
        if y_prob is None:
            raise ConfigValidationError(["Для калибровки нужен columns.score_column (вероятности)."])
        # Калибровочная кривая
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy='uniform'
        )
        
        # Brier score для оценки калибровки
        brier_score = brier_score_loss(y_true, y_prob)
        
        # График калибровки
        plt.figure(figsize=(12, 8))  # Стандартный размер
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=f'Model (Brier Score: {brier_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot (Reliability Diagram)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Создание таблицы с результатами по бакетам
        bins_df = pd.DataFrame({
            'Bin': range(1, len(mean_predicted_value) + 1),
            'Mean_Predicted': mean_predicted_value,
            'Fraction_Positive': fraction_of_positives,
            'Difference': np.abs(mean_predicted_value - fraction_of_positives)
        })
        
        html_content = f"""
        <h4>Model Calibration Test</h4>
        <p><strong>Brier Score:</strong> {brier_score:.4f} (чем меньше, тем лучше)</p>
        <img src="data:image/png;base64,{plot_img}" />
        <h5>Calibration by Bins:</h5>
        {bins_df.to_html(index=False, float_format="%.3f")}
        <p><strong>Интерпретация:</strong> Чем ближе кривая калибровки к диагонали, тем лучше калиброванность модели.
        Brier Score < 0.25 считается хорошим результатом.</p>
        """
        
        return {"report": html_content}


class M44_OverfittingTest(BaseModelTest):
    """
    Тест 4.2.1. Наличие переобучения
    """
    def __init__(self, model_type: str = 'classification'):
        super().__init__("M 4.4", "Overfitting Detection")
        self.model_type = model_type

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, _ = _split_frames_from_cfg(df, cfg)

        if self.model_type == 'classification':
            y_tr, yhat_tr, _ = _get_y_cols(df_train, cfg, for_classification=True)
            y_te, yhat_te, _ = _get_y_cols(df_test,  cfg, for_classification=True)
            train_mean = f1_score(y_tr, yhat_tr)
            test_mean  = f1_score(y_te, yhat_te)
        else:
            y_tr, _, s_tr = _get_y_cols(df_train, cfg, for_classification=False)
            y_te, _, s_te = _get_y_cols(df_test,  cfg, for_classification=False)
            train_mean = r2_score(y_tr, s_tr)
            test_mean  = r2_score(y_te, s_te)

        relative_change = ((test_mean - train_mean) / train_mean) * 100 if train_mean != 0 else 0.0
        if abs(relative_change) > 20:   signal = "red"
        elif abs(relative_change) > 10: signal = "orange"
        else:                           signal = "green"
        self.test_signal = signal
        
        results_df = pd.DataFrame({
            'Metric': ['Train Score', 'Test Score', 'Relative Change, %'],
            'Value': [f"{train_mean:.3f}", f"{test_mean:.3f}", f"{relative_change:.1f}%"]
        })
        
        html_content = f"""
        <h4>Overfitting Detection Test</h4>
        <p><strong>Signal:</strong> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        {results_df.to_html(index=False)}
        <p><strong>Интерпретация:</strong></p>
        <ul>
            <li>Зеленый: |относительное изменение| ≤ 10%</li>
            <li>Оранжевый: 10% < |относительное изменение| ≤ 20%</li>
            <li>Красный: |относительное изменение| > 20%</li>
        </ul>
        """
        
        self.test_signal = signal
        return {"report": html_content}


class M45_MetricStabilityTest(BaseModelTest):
    """
    Тест 4.2.2. Стабильность метрики в динамике
    """
    def __init__(self):
        super().__init__("M 4.5", "Metric Stability Over Time")

    def run(self, df: pd.DataFrame, config: dict, sample: str = 'test', freq: str = 'M', confidence_intervals: Optional[Dict] = None) -> Dict[str, str]:
        cfg = validate_config(config, df)
        date_col = cfg.columns.date_column
        df_train, df_test, df_gen = _split_frames_from_cfg(df, cfg)
        df_used = {"train": df_train, "test": df_test, "genpop": df_gen}.get(str(sample).lower(), df_test).copy()
        df_used[date_col] = pd.to_datetime(df_used[date_col], errors='coerce')

        # считаем метрику по периодам
        if cfg.task == 'classification':
            def _metric(g):
                y, yhat, _ = _get_y_cols(g, cfg, for_classification=True)
                return f1_score(y, yhat)
        else:
            def _metric(g):
                y, _, s = _get_y_cols(g, cfg, for_classification=False)
                return r2_score(y, s)

        scores = (df_used.dropna(subset=[date_col])
                        .sort_values(date_col)
                        .groupby(pd.Grouper(key=date_col, freq=freq))
                        .apply(_metric)
                        .dropna())
        dates = scores.index.to_series()
        plt.figure(figsize=(12, 8))  # Стандартный размер
        plt.plot(dates, scores, 'b-o', label='Metric Score')
        
        # Добавление доверительных интервалов если есть
        if confidence_intervals:
            lower_bound = confidence_intervals.get('lower', scores.min())
            upper_bound = confidence_intervals.get('upper', scores.max())
            plt.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.7, label='Lower CI')
            plt.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.7, label='Upper CI')
        
        plt.xlabel('Date')
        plt.ylabel('Metric Score')
        plt.title('Metric Stability Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Расчет волатильности и определение сигнала
        volatility = scores.std()
        mean_score = scores.mean()
        cv = volatility / mean_score if mean_score != 0 else 0
        
        # Определение сигнала светофора на основе коэффициента вариации
        if cv <= 0.05:  # Низкая волатильность
            self.test_signal = "green"
        elif cv <= 0.15:  # Умеренная волатильность 
            self.test_signal = "orange"
        else:  # Высокая волатильность
            self.test_signal = "red"
        
        signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>', "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>', "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]
        
        html_content = f"""
        <h4>Metric Stability Over Time {signal_color}</h4>
        <p><strong>Mean Score:</strong> {mean_score:.4f}</p>
        <p><strong>Standard Deviation:</strong> {volatility:.4f}</p>
        <p><strong>Coefficient of Variation:</strong> {cv:.4f}</p>
        <img src="data:image/png;base64,{plot_img}" />
        <p><strong>Interpretation:</strong> This test is informative in nature. 
        High volatility may indicate model instability.</p>
        <p><strong>Signal:</strong> {signal_color} - {'Stable metric' if self.test_signal == 'green' else 'Moderate volatility' if self.test_signal == 'orange' else 'High volatility detected'}</p>
        """
        
        return {"report": html_content}


class M46_CategoryQualityTest(BaseModelTest): #NOT FOR USE
    """
    Тест 4.2.3. Качество в разрезе категорий
    """
    def __init__(self, model_type: str = 'classification'):
        super().__init__("M 4.6", "Quality by Categories")
        self.model_type = model_type

    def run(self, df: pd.DataFrame, category_column: str, target_column: str, 
            pred_column: str, overall_ci: Optional[Dict] = None) -> Dict[str, str]:
        """
        Args:
            df (pd.DataFrame): Данные
            category_column (str): Категориальный признак для группировки
            target_column (str): Целевая переменная
            pred_column (str): Предсказания модели
            overall_ci (Dict, optional): Общий доверительный интервал из теста 4.1.1
        """
        results = []
        
        # Общая метрика
        if self.model_type == 'classification':
            overall_metric = f1_score(df[target_column], df[pred_column])
        else:
            overall_metric = r2_score(df[target_column], df[pred_column])
        
        results.append({
            'Category': 'Overall',
            'Metric': overall_metric,
            'Sample_Size': len(df),
            'Percentage': 100.0
        })
        
        # Метрики по категориям
        for category in df[category_column].unique():
            if pd.isna(category):
                continue
                
            category_data = df[df[category_column] == category]
            
            if len(category_data) < 10:  # Минимальный размер для анализа
                continue
            
            if self.model_type == 'classification':
                metric = f1_score(category_data[target_column], category_data[pred_column])
            else:
                metric = r2_score(category_data[target_column], category_data[pred_column])
            
            results.append({
                'Category': str(category),
                'Metric': metric,
                'Sample_Size': len(category_data),
                'Percentage': (len(category_data) / len(df)) * 100
            })
        
        results_df = pd.DataFrame(results)
        
        # Создание графика
        plt.figure(figsize=(12, 8))  # Стандартный размер
        categories = results_df['Category'][1:]  # Исключаем Overall
        metrics = results_df['Metric'][1:]
        
        plt.bar(categories, metrics, alpha=0.7)
        plt.axhline(y=overall_metric, color='r', linestyle='-', label=f'Overall: {overall_metric:.3f}')
        
        if overall_ci:
            plt.axhline(y=overall_ci.get('lower', overall_metric), color='r', 
                       linestyle='--', alpha=0.7, label='CI Lower')
            plt.axhline(y=overall_ci.get('upper', overall_metric), color='r', 
                       linestyle='--', alpha=0.7, label='CI Upper')
        
        plt.xlabel('Category')
        plt.ylabel('Metric Score')
        plt.title(f'Model Quality by {category_column}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Определение сигнала светофора на основе отклонения от общей метрики
        if len(results_df) > 1:
            category_metrics = results_df['Metric'][1:]  # Исключаем Overall
            deviation_from_overall = np.abs(category_metrics - overall_metric).max()
            relative_deviation = deviation_from_overall / abs(overall_metric) if overall_metric != 0 else 1
            
            if relative_deviation <= 0.1:  # Низкое отклонение
                self.test_signal = "green"
            elif relative_deviation <= 0.25:  # Умеренное отклонение
                self.test_signal = "orange"
            else:  # Высокое отклонение
                self.test_signal = "red"
        else:
            self.test_signal = "orange"
        
        signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>', "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>', "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]
        
        html_content = f"""
        <h4>Quality by Categories Test {signal_color}</h4>
        <img src="data:image/png;base64,{plot_img}" />
        <h5>Results by Category:</h5>
        {results_df.to_html(index=False, float_format="%.3f")}
        <p><strong>Interpretation:</strong> This test is informative in nature. 
        Metrics in groups are expected to be close to the overall metric and fall within the confidence interval.</p>
        <p><strong>Signal:</strong> {signal_color} - {'Consistent quality across categories' if self.test_signal == 'green' else 'Moderate quality variation' if self.test_signal == 'orange' else 'Significant quality differences detected'}</p>
        """
        
        return {"report": html_content}


class M47_ShapImportanceTest(BaseModelTest):
    """
    Section 4.3.1. SHAP importance comparison between training and validation
    """
    def __init__(self):
        super().__init__("M 4.7", "SHAP Importance Comparison")

    def run(self, df: pd.DataFrame, config: dict, model, top_n: int = 20) -> Dict[str, str]:
        cfg = validate_config(config, df)
        feats = _features_numeric(cfg)
        df_train, df_test, _ = _split_frames_from_cfg(df, cfg)
        X_train = df_train[feats].copy()
        X_test  = df_test[feats].copy()
        if not SHAP_AVAILABLE:
            return {"report": "<h4>SHAP Importance Test</h4><p>SHAP library is not available. Please install it.</p>"}
        
        # WARNING: May take very long time for large datasets and complex models
        try:
            # Calculate SHAP values
            explainer = shap.Explainer(model, X_train.sample(min(100, len(X_train))))
            shap_values_train = explainer(X_train.sample(min(500, len(X_train))))
            shap_values_test = explainer(X_test.sample(min(500, len(X_test))))
            
            # Mean absolute SHAP importance
            shap_importance_train = np.abs(shap_values_train.values).mean(0)
            shap_importance_test = np.abs(shap_values_test.values).mean(0)
            
            # Create results dataframe
            results = []
            for i, feature in enumerate(X_train.columns):
                train_imp = shap_importance_train[i]
                test_imp = shap_importance_test[i]
                diff = test_imp - train_imp
                diff_pct = (diff / train_imp * 100) if train_imp != 0 else 0
                
                results.append({
                    'Feature': feature,
                    'SHAP_imp_train': train_imp,
                    'SHAP_imp_test': test_imp,
                    'SHAP_imp_diff': diff,
                    'SHAP_imp_diff_%': diff_pct
                })
            
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('SHAP_imp_train', ascending=False).head(top_n)
            
            # Calculate ratio of features with >50% change
            high_change_features = (np.abs(results_df['SHAP_imp_diff_%']) > 50).sum()
            total_features = len(results_df)
            high_change_ratio = high_change_features / total_features
            
            # Determine traffic light signal
            if high_change_ratio < 0.3:
                signal = "green"
            elif high_change_ratio < 0.5:
                signal = "orange"
            else:
                signal = "red"
            
            # Create SHAP plots using SHAP library
            plots_html = ""
            
            # 1. SHAP Summary plot for train
            plt.figure(figsize=(12, 8))  # Стандартный размер
            shap.plots.beeswarm(shap_values_train, max_display=min(15, len(X_train.columns)), show=False)
            plt.title('SHAP Summary Plot - Training Data')
            buf1 = BytesIO()
            plt.savefig(buf1, format="png", bbox_inches='tight', dpi=150)
            plt.close()
            plot1_img = base64.b64encode(buf1.getvalue()).decode("utf-8")
            
            # 2. SHAP Summary plot for test
            plt.figure(figsize=(12, 8))  # Стандартный размер
            shap.plots.beeswarm(shap_values_test, max_display=min(15, len(X_test.columns)), show=False)
            plt.title('SHAP Summary Plot - Test Data')
            buf2 = BytesIO()
            plt.savefig(buf2, format="png", bbox_inches='tight', dpi=150)
            plt.close()
            plot2_img = base64.b64encode(buf2.getvalue()).decode("utf-8")
            
            # 3. SHAP Bar plot comparison
            plt.figure(figsize=(12, 8))  # Уже правильный размер
            x_pos = np.arange(len(results_df))
            width = 0.35
            plt.bar(x_pos - width/2, results_df['SHAP_imp_train'], width, alpha=0.8, label='Train', color='skyblue')
            plt.bar(x_pos + width/2, results_df['SHAP_imp_test'], width, alpha=0.8, label='Test', color='lightcoral')
            plt.xticks(x_pos, results_df['Feature'], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Mean |SHAP Value|')
            plt.title(f'Top {len(results_df)} Features: SHAP Importance Comparison')
            plt.legend()
            plt.tight_layout()
            
            buf3 = BytesIO()
            plt.savefig(buf3, format="png", bbox_inches='tight', dpi=150)
            plt.close()
            plot3_img = base64.b64encode(buf3.getvalue()).decode("utf-8")
            
            html_content = f"""
            <h4>SHAP Importance Comparison Test</h4>
            <p><strong>Signal:</strong> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
            <p><strong>Features with >50% change:</strong> {high_change_features}/{total_features} ({high_change_ratio:.1%})</p>
            
            <h5>SHAP Summary Plot - Training Data</h5>
            <img src="data:image/png;base64,{plot1_img}" />
            
            <h5>SHAP Summary Plot - Test Data</h5>
            <img src="data:image/png;base64,{plot2_img}" />
            
            <h5>SHAP Importance Comparison</h5>
            <img src="data:image/png;base64,{plot3_img}" />
            
            <h5>Top {len(results_df)} Features Detailed Comparison:</h5>
            {results_df.to_html(index=False, float_format="%.4f")}
            """
            
            self.test_signal = signal
            return {"report": html_content}
            
        except Exception as e:
            self.test_signal = "red"
            return {"report": f"<h4>SHAP Importance Test</h4><p>Error calculating SHAP values: {str(e)}</p>"}


class M48_FeatureContributionTest(BaseModelTest):
    def __init__(self):
        super().__init__("M 4.8", "Feature Contribution Adequacy")

    def run(self,
            df: pd.DataFrame = None,
            config: dict = None,
            model=None,
            sample: str = "test",
            feature_descriptions: Dict[str, str] | None = None,
            X_sample: pd.DataFrame | None = None  # backward compat
            ) -> Dict[str, str]:

        # --- Backward compatibility ---
        if df is None and X_sample is not None and model is not None and config is None:
            # старый путь: оставляем прежнюю реализацию на X_sample
            X_use = X_sample.copy()
        else:
            if df is None or config is None or model is None:
                raise ValueError("Please provide df, config and model.")
            sample_col = _cfg_col(config, "sample_column")
            # берём нужный срез и признаки
            df_s = df if sample is None else df[df[sample_col] == sample]
            X_use = _select_X(df_s, config)

        if hasattr(model, "feature_names_in_"):
            # sklearn >= 1.0: используем порядок/подмножество обучающих фич
            use_feats = [f for f in model.feature_names_in_ if f in X_use.columns]
            if not use_feats:
                raise ValueError("None of model.feature_names_in_ are present in the provided DataFrame.")
            X_use = X_use[use_feats]
        else:
            # на всякий случай ограничим численными фичами
            X_use = _numeric_only(X_use)

        if not SHAP_AVAILABLE:
            return {"report": "<h4>Feature Contribution Test</h4><p>SHAP library is not available. Please install it.</p>"}

        try:
            # ограничим размер для скорости
            X_bg = X_use.sample(min(50, len(X_use)), random_state=42)
            X_eval = X_use.sample(min(100, len(X_use)), random_state=42)

            explainer = shap.Explainer(model, X_bg)
            shap_values = explainer(X_eval)

            mean_shap = np.mean(shap_values.values, axis=0)
            contributions_df = pd.DataFrame({
                'Feature': X_use.columns,
                'Mean_SHAP_Value': mean_shap,
                'Abs_Mean_SHAP': np.abs(mean_shap)
            }).sort_values('Abs_Mean_SHAP', ascending=False)

            top_10_importance = contributions_df.head(10)['Abs_Mean_SHAP'].sum()
            total_importance = contributions_df['Abs_Mean_SHAP'].sum()
            concentration_ratio = top_10_importance / total_importance if total_importance > 0 else 1.0

            if concentration_ratio < 0.8:
                self.test_signal = "green"
            elif concentration_ratio < 0.95:
                self.test_signal = "orange"
            else:
                self.test_signal = "red"

            signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>',
                            "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>',
                            "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]

            html_content = f"""
            <h4>Feature Contribution Adequacy Test {signal_color}</h4>
            <p>Analysis of feature contributions to the model. This test is informative in nature.</p>
            <h5>Feature Contributions (Top 15):</h5>
            {contributions_df.head(15).to_html(index=False, float_format="%.4f")}
            <p><strong>Interpretation:</strong> Analyze whether the signs and magnitudes of contributions 
            correspond to business logic and common sense.</p>
            <p><strong>Signal:</strong> {signal_color} - {'Balanced feature importance' if self.test_signal == 'green' else 'Moderate concentration' if self.test_signal == 'orange' else 'High concentration in few features'}</p>
            """

            if feature_descriptions:
                html_content += "<h5>Feature Descriptions:</h5><ul>"
                for feature in contributions_df.head(10)['Feature']:
                    desc = feature_descriptions.get(feature, "No description")
                    html_content += f"<li><strong>{feature}:</strong> {desc}</li>"
                html_content += "</ul>"

            return {"report": html_content}
        except Exception as e:
            self.test_signal = "red"
            return {"report": f"<h4>Feature Contribution Test</h4><p>Error: {str(e)}</p>"}

class M49_UpliftTest(BaseModelTest):
    def __init__(self, model_type: str = 'classification', step: int = 1):
        super().__init__("M 4.9", "Feature Uplift Test")
        self.model_type = model_type
        self.step = step

    def run(self,
            df: pd.DataFrame = None,
            config: dict = None,
            model_class=None,
            feature_importance: pd.Series | None = None,
            X_train: pd.DataFrame | None = None, y_train: pd.Series | None = None,
            X_test: pd.DataFrame | None = None, y_test: pd.Series | None = None
            ) -> Dict[str, str]:

        # --- Backward compatibility ---
        if df is None and all(v is not None for v in [X_train, y_train, X_test, y_test, model_class]) and config is None:
            feats_order = list(feature_importance.index) if isinstance(feature_importance, pd.Series) else list(X_train.columns)
        else:
            if df is None or config is None or model_class is None:
                raise ValueError("Please provide df, config and model_class.")
            sample_col = _cfg_col(config, "sample_column")
            target_col = _cfg_col(config, "target_column")

            df_tr = _subset_by_sample(df, config, "train")
            df_te = _subset_by_sample(df, config, "test")
            X_train = _select_X(df_tr, config)
            X_test  = _select_X(df_te, config)
            y_train = df_tr[target_col]
            y_test  = df_te[target_col]

            # порядок признаков: если передали — используем, иначе — учим модель и берём feature_importances_
            if isinstance(feature_importance, pd.Series):
                feats_order = [f for f in feature_importance.sort_values(ascending=False).index if f in X_train.columns]
            else:
                base_model = model_class()
                base_model.fit(X_train, y_train)
                if hasattr(base_model, "feature_importances_"):
                    importances = pd.Series(base_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                    feats_order = list(importances.index)
                else:
                    # fallback: как пришли
                    feats_order = list(X_train.columns)

        # основной цикл добавления признаков
        results = []
        best_test = -np.inf
        optimal_features = 0

        for k in range(self.step, len(feats_order) + 1, self.step):
            use_feats = feats_order[:k]
            model = model_class()
            model.fit(X_train[use_feats], y_train)

            if self.model_type == "classification":
                y_pred = model.predict(X_test[use_feats])
                score = f1_score(y_test, y_pred)
            else:
                y_pred = model.predict(X_test[use_feats])
                score = r2_score(y_test, y_pred)

            results.append((k, score))
            if score > best_test:
                best_test = score
                optimal_features = k

        res_df = pd.DataFrame(results, columns=["#Features", "TestScore"])
        max_test_score = res_df["TestScore"].max()

        # светофор
        if len(res_df) >= 3 and res_df["TestScore"].iloc[-1] < max_test_score * 0.98:
            signal = "red"      # явное ухудшение к концу — лишние фичи
        elif len(res_df) >= 3 and res_df["TestScore"].iloc[-1] < max_test_score * 0.995:
            signal = "orange"   # лёгкое ухудшение
        else:
            signal = "green"

        # график
        plt.figure(figsize=(12, 8))
        plt.plot(res_df["#Features"], res_df["TestScore"], marker="o")
        plt.axvline(optimal_features, linestyle="--")
        plt.xlabel("#Features")
        plt.ylabel("Score")
        plt.title("Feature Uplift Curve")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        self.test_signal = signal
        html_content = f"""
        <h4>Feature Uplift Test</h4>
        <p><strong>Signal:</strong> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        <p><strong>Optimal number of features:</strong> {optimal_features}</p>
        <p><strong>Max test score:</strong> {max_test_score:.4f}</p>
        <img src="data:image/png;base64,{plot_img}" />
        <p><strong>Интерпретация:</strong> Если кривая test выходит на плато или начинает снижаться, 
        это может указывать на наличие избыточных признаков.</p>
        """
        return {"report": html_content}


class M410_MissingValuesImpactTest(BaseModelTest):
    """
    Тест 4.10. Определение признаков с большим количеством пропущенных значений
    """
    def __init__(self, threshold=0.8):
        super().__init__('M 4.10', 'Features with High Missing Value Ratio')
        self.threshold = threshold

    def run(self,
            df: pd.DataFrame = None,
            config: dict = None,
            X: pd.DataFrame = None,              # legacy-режим (обратная совместимость)
            time_column: str | None = None       # legacy-режим
            ) -> Dict[str, str]:
        """
        Новый контракт: передавайте df + config.
        Legacy: X + time_column продолжает работать.
        """
        import io, base64
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # --- Новый путь: df + config ---
        if df is not None and config is not None:
            # поддержка вложенного вида config["columns"]
            cols = config["columns"] if isinstance(config, dict) and "columns" in config else config

            # соберём список фич из конфига
            num = cols.get("numeric_features", []) or []
            cat = cols.get("categorical_features", []) or []
            features = list(dict.fromkeys([*num, *cat]))  # порядок + без дублей

            # исключим служебные колонки
            reserved = {
                cols.get("date_column"),
                cols.get("target_column"),
                cols.get("sample_column"),
                cols.get("id_column"),
                cols.get("score_column"),
                cols.get("prediction_column"),
            }
            features = [f for f in features if f and f not in reserved]

            if not features:
                raise ValueError("Пустой список признаков: задайте numeric_features и/или categorical_features в config.columns")

            missing = [c for c in features if c not in df.columns]
            if missing:
                raise KeyError(f"В DataFrame отсутствуют колонки из списка фич: {missing}")

            X_use = df[features].copy()
            time_col = cols.get("date_column", None)

            # приведём дату к datetime для временных графиков
            if time_col and time_col in df.columns:
                try:
                    df = df.copy()
                    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                except Exception:
                    pass

        # --- Legacy путь: X + time_column ---
        elif X is not None:
            X_use = X.copy()
            time_col = time_column
            # в legacy режиме у нас может не быть df; временная динамика будет недоступна, если time_col не в X_use
            if time_col is not None and time_col not in X_use.columns:
                # ок, просто игнорируем динамику
                time_col = None
        else:
            raise ValueError("Передайте либо (df, config), либо (X, time_column)")

        # ===== 1) Доля пропусков по признакам =====
        feature_columns = [c for c in X_use.columns if c != time_col]  # не анализируем саму колонку времени
        X_features = X_use[feature_columns]

        missing_ratios = X_features.isnull().mean().sort_values(ascending=False)
        high_missing = missing_ratios[missing_ratios > self.threshold]

        # ===== 2) Визуализация: топ-30 признаков по доле пропусков (barh) =====
        plots_html = []
        if len(missing_ratios) > 0:
            top = missing_ratios.head(30)[::-1]  # для горизонтального барчарта снизу вверх
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(top.index, top.values)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Missing ratio")
            ax.set_title("Missing values ratio by feature (top 30)")
            ax.grid(True, linestyle="--", linewidth=0.5, axis="x")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            img_bar = base64.b64encode(buf.getvalue()).decode("utf-8")
            plots_html.append(f'<h5>Top-30 missing ratios</h5><img src="data:image/png;base64,{img_bar}">')

        # ===== 3) Таблица признаков с высокой долей пропусков =====
        table_html = "<p><i>Нет признаков с долей пропусков выше порога.</i></p>"
        if len(high_missing) > 0:
            tbl = high_missing.to_frame(name="MissingRatio").reset_index().rename(columns={"index": "Feature"})
            tbl["MissingRatio"] = (tbl["MissingRatio"] * 100).round(2)
            table_html = tbl.to_html(index=False)

        # ===== 4) Динамика пропусков по времени (heatmap для high_missing) =====
        heatmap_html = ""
        if time_col is not None and time_col in getattr(df, "columns", []) and len(high_missing) > 0:
            # соберём фрейм: время + интересующие признаки
            tmp = pd.concat([df[[time_col]].reset_index(drop=True),
                            X_use[high_missing.index].reset_index(drop=True)], axis=1)
            tmp = tmp.dropna(subset=[time_col])
            if len(tmp) > 0:
                # месячная агрегация по умолчанию
                tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                tmp = tmp.dropna(subset=[time_col])
                grouped = tmp.groupby(pd.Grouper(key=time_col, freq="M"))
                # матрица: периоды x признаки (доля пропусков)
                periods = []
                rows = []
                for period, g in grouped:
                    if period is pd.NaT:
                        continue
                    periods.append(period)
                    rows.append(g[high_missing.index].isnull().mean().values)
                if len(rows) > 0:
                    M = np.vstack(rows)  # shape: (#periods, #features)
                    # строим heatmap (features x periods)
                    fig2, ax2 = plt.subplots(figsize=(max(10, len(periods) * 0.4),
                                                    max(4, len(high_missing) * 0.5)))
                    im = ax2.imshow(M.T, aspect="auto", origin="lower", vmin=0, vmax=1)
                    ax2.set_yticks(np.arange(len(high_missing.index)))
                    ax2.set_yticklabels(high_missing.index)
                    ax2.set_xticks(np.arange(len(periods)))
                    ax2.set_xticklabels([p.strftime("%Y-%m") for p in periods],
                                        rotation=45, ha="right")
                    ax2.set_title("Missing ratio over time (features with high missing)")
                    ax2.grid(False)
                    cbar = fig2.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
                    cbar.set_label("Missing ratio")
                    plt.tight_layout()

                    buf2 = io.BytesIO()
                    plt.savefig(buf2, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig2)
                    img_heat = base64.b64encode(buf2.getvalue()).decode("utf-8")
                    heatmap_html = f'<h5>Dynamics heatmap (monthly)</h5><img src="data:image/png;base64,{img_heat}">'

        # ===== 5) Сигнал по доле «плохих» признаков =====
        share_high = (len(high_missing) / max(len(feature_columns), 1)) if len(feature_columns) else 0.0
        if share_high == 0:
            signal = "green"
        elif share_high <= 0.10:
            signal = "yellow"
        else:
            signal = "red"
        self.test_signal = signal

        # ===== 6) Сборка HTML =====
        header = f"""
        <h4>Features with High Missing Value Ratio (threshold = {int(self.threshold*100)}%)</h4>
        <p><b>Signal:</b> <span style="color:{signal}; font-weight:bold">{signal.upper()}</span></p>
        <p>Всего фич (без time): <b>{len(feature_columns)}</b>. 
        Фич с пропусками &gt; {int(self.threshold*100)}%: <b>{len(high_missing)}</b> 
        ({round(share_high*100, 2)}%).</p>
        """

        html = header + "".join(plots_html) + "<h5>Features above threshold</h5>" + table_html + heatmap_html
        return {"DASHBOARD": html}


class M411_TargetCorrelationTest(BaseModelTest):
    """
    Тест 4.4.1. Корреляция признаков с таргетом
    """
    def __init__(self):
        super().__init__("M 4.11", "Feature-Target Correlation")

    def run(self, X: pd.DataFrame, y: pd.Series, top_features: List[str] = None) -> Dict[str, str]:
        """
        Args:
            X (pd.DataFrame): Признаки
            y (pd.Series): Целевая переменная
            top_features (List[str], optional): Топ важные признаки для анализа
        """
        correlations = []
        
        # Анализируем корреляцию для числовых признаков
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        if top_features:
            features_to_analyze = [f for f in top_features if f in numeric_features]
        else:
            features_to_analyze = numeric_features[:20]  # Топ 20 числовых признаков
        
        for feature in features_to_analyze:
            try:
                corr, p_value = pearsonr(X[feature].fillna(X[feature].median()), y)
                correlations.append({
                    'Feature': feature,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except:
                correlations.append({
                    'Feature': feature,
                    'Correlation': 0,
                    'P_Value': 1,
                    'Significant': 'Error'
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
        
        # График корреляций
        plt.figure(figsize=(12, 8))  # Стандартный размер
        plt.barh(range(len(corr_df)), corr_df['Correlation'])
        plt.yticks(range(len(corr_df)), corr_df['Feature'])
        plt.xlabel('Correlation with Target')
        plt.title('Feature-Target Correlations')
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Определение сигнала светофора на основе высоких корреляций
        high_correlations = corr_df[abs(corr_df['Correlation']) > 0.7]
        max_correlation = abs(corr_df['Correlation']).max() if len(corr_df) > 0 else 0
        
        if max_correlation < 0.5:  # Низкие корреляции
            self.test_signal = "green"
        elif max_correlation < 0.8:  # Умеренные корреляции
            self.test_signal = "orange"
        else:  # Высокие корреляции (возможные утечки)
            self.test_signal = "red"
        
        signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>', "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>', "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]
        
        html_content = f"""
        <h4>Feature-Target Correlation Test {signal_color}</h4>
        <p>Analysis of linear correlation between features and target variable.</p>
        <img src="data:image/png;base64,{plot_img}" />
        <h5>Correlation Results:</h5>
        {corr_df.to_html(index=False, float_format="%.4f")}
        <p><strong>Interpretation:</strong> This test is informative in nature. 
        High correlations (|r| > 0.7) may indicate potential data leaks.</p>
        <p><strong>Signal:</strong> {signal_color} - {'Normal correlations' if self.test_signal == 'green' else 'Moderate correlations found' if self.test_signal == 'orange' else 'High correlations detected - check for leaks'}</p>
        """
        
        return {"report": html_content}


class M412_FeatureCorrelationTest(BaseModelTest):
    """
    Тест 4.4.2. Корреляция признаков между собой
    """
    def __init__(self, threshold: float = 0.95):
        super().__init__("M 4.12", "Feature-Feature Correlation")
        self.threshold = threshold

    def run(self, X: pd.DataFrame, top_features: List[str] = None) -> Dict[str, str]:
        """
        Args:
            X (pd.DataFrame): Признаки
            top_features (List[str], optional): Топ важные признаки для анализа
        """
        # Выбираем только числовые признаки
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        if top_features:
            features_to_analyze = [f for f in top_features if f in numeric_features]
        else:
            features_to_analyze = numeric_features[:20]  # Топ 20 признаков
        
        X_subset = X[features_to_analyze].fillna(X[features_to_analyze].median())
        
        # Матрица корреляций
        corr_matrix = X_subset.corr()
        
        # Поиск высоко коррелированных пар
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > self.threshold:
                    high_corr_pairs.append({
                        'Feature_1': corr_matrix.columns[i],
                        'Feature_2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        # Создание тепловой карты
        plt.figure(figsize=(12, 12))  # Квадратная heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        html_content = f"""
        <h4>Feature-Feature Correlation Test</h4>
        <p><strong>Threshold:</strong> {self.threshold}</p>
        <p><strong>High correlation pairs found:</strong> {len(high_corr_pairs)}</p>
        <img src="data:image/png;base64,{plot_img}" />
        """
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            html_content += f"""
            <h5>High Correlation Pairs (|r| > {self.threshold}):</h5>
            {high_corr_df.to_html(index=False, float_format="%.4f")}
            """
        
        # Определение сигнала светофора на основе количества высоко коррелированных пар
        if len(high_corr_pairs) == 0:
            self.test_signal = "green"
        elif len(high_corr_pairs) <= 2:
            self.test_signal = "orange"
        else:
            self.test_signal = "red"
        
        signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>', "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>', "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]
        
        html_content += f"""
        <p><strong>Interpretation:</strong> High correlations between features may indicate 
        redundancy and the need to remove one of the correlated features.</p>
        <p><strong>Signal:</strong> {signal_color} - {'No high correlations found' if self.test_signal == 'green' else 'Few high correlations detected' if self.test_signal == 'orange' else 'Many high correlations found'}</p>
        """
        
        return {"report": html_content}


class M413_VIFTest(BaseModelTest):
    """
    Тест 4.4.3. VIF (Variance Inflation Factor)
    """
    def __init__(self):
        super().__init__("M 4.13", "VIF Analysis")

    def run(self, X: pd.DataFrame, top_features: List[str] = None) -> Dict[str, str]:
        """
        Args:
            X (pd.DataFrame): Признаки
            top_features (List[str], optional): Топ важные признаки для анализа
        """
        try:
            # Выбираем только числовые признаки
            numeric_features = X.select_dtypes(include=[np.number]).columns
            
            if top_features:
                features_to_analyze = [f for f in top_features if f in numeric_features]
            else:
                features_to_analyze = numeric_features[:15]  # Ограничиваем для производительности
            
            X_subset = X[features_to_analyze].fillna(X[features_to_analyze].median())
            
            # Удаляем строки с пропусками
            X_clean = X_subset.dropna()
            
            if len(X_clean) < 50:
                return {"report": "<h4>VIF Analysis</h4><p>Insufficient data for VIF calculation (too many missing values).</p>"}
            
            # Расчет VIF
            vif_data = []
            for i, feature in enumerate(X_clean.columns):
                try:
                    vif_value = variance_inflation_factor(X_clean.values, i)
                    vif_data.append({
                        'Feature': feature,
                        'VIF': vif_value,
                        'Status': 'High' if vif_value > 10 else 'Moderate' if vif_value > 5 else 'Low'
                    })
                except:
                    vif_data.append({
                        'Feature': feature,
                        'VIF': float('inf'),
                        'Status': 'Error'
                    })
            
            vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
            
            # График VIF
            plt.figure(figsize=(12, 8))  # Стандартный размер
            plt.barh(range(len(vif_df)), vif_df['VIF'])
            plt.yticks(range(len(vif_df)), vif_df['Feature'])
            plt.xlabel('VIF Value')
            plt.title('Variance Inflation Factor by Feature')
            plt.axvline(x=5, color='orange', linestyle='--', label='Moderate (VIF=5)')
            plt.axvline(x=10, color='red', linestyle='--', label='High (VIF=10)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            high_vif_count = (vif_df['VIF'] > 10).sum()
            moderate_vif_count = ((vif_df['VIF'] >= 5) & (vif_df['VIF'] <= 10)).sum()
            
            # Определение сигнала светофора на основе VIF
            if high_vif_count == 0 and moderate_vif_count <= 2:
                self.test_signal = "green"
            elif high_vif_count <= 1 or moderate_vif_count <= 5:
                self.test_signal = "orange"
            else:
                self.test_signal = "red"
            
            signal_color = {"green": '<span style="color: green; font-weight: bold;">●</span>', "orange": '<span style="color: #B8860B; font-weight: bold;">●</span>', "red": '<span style="color: red; font-weight: bold;">●</span>'}[self.test_signal]
            
            html_content = f"""
            <h4>VIF Analysis {signal_color}</h4>
            <p><strong>Features with high VIF (>10):</strong> {high_vif_count}</p>
            <img src="data:image/png;base64,{plot_img}" />
            <h5>VIF Results:</h5>
            {vif_df.to_html(index=False, float_format="%.2f")}
            <p><strong>Interpretation:</strong></p>
            <ul>
                <li>VIF < 5: Low multicollinearity</li>
                <li>5 ≤ VIF < 10: Moderate multicollinearity</li>
                <li>VIF ≥ 10: High multicollinearity (feature removal recommended)</li>
            </ul>
            <p><strong>Signal:</strong> {signal_color} - {'Low multicollinearity' if self.test_signal == 'green' else 'Moderate multicollinearity detected' if self.test_signal == 'orange' else 'High multicollinearity found'}</p>
            """
            
            return {"report": html_content}
            
        except Exception as e:
            return {"report": f"<h4>VIF Analysis</h4><p>Error calculating VIF: {str(e)}</p>"}

class M414_TwoForestSelectionTest(BaseModelTest):
    """
    M 4.14 — Two-Forest Feature Selection
    Идея: обучаем два случайных леса с разными random_state на train, сравниваем
    стабильность отбора топ-K признаков и важности.
    """
    def __init__(self,
                 model_type: str = "classification",
                 top_k: int = 20,
                 n_estimators: int = 300,
                 max_depth=None,
                 random_state1: int = 42,
                 random_state2: int = 777):
        super().__init__("M 4.14", "Two-Forest Feature Selection")
        self.model_type = model_type
        self.top_k = int(top_k)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state1 = int(random_state1)
        self.random_state2 = int(random_state2)

    def _split_frames(self, df: pd.DataFrame, sample_col: str):
        s = df[sample_col].astype(str).str.lower()
        df_train = df[s.eq("train")]
        df_test  = df[s.eq("test")]
        return df_train, df_test

    def _prep_xy(self, df: pd.DataFrame, features: list, target_col: str):
        X = df[features].copy()
        # numeric only (на случай попадания категориальных)
        X = X.apply(pd.to_numeric, errors="coerce")
        # простая обработка пропусков для обучения
        if X.isnull().any().any():
            X = X.fillna(X.median(numeric_only=True))
        y = df[target_col].copy()
        if y.isnull().any():
            # мягкая импутация: для clf — мода, для reg — медиана
            y = y.fillna(y.mode().iloc[0] if self.model_type == "classification" else y.median())
        return X, y

    def _make_model(self, rs: int):
        if self.model_type == "classification":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=rs,
                n_jobs=-1
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=rs,
                n_jobs=-1
            )

    def run(self, df: pd.DataFrame, config: dict) -> dict:
        # 1) валидируем конфиг и собираем данные
        cfg = validate_config(config, df)
        cols = config["columns"] if "columns" in config else config

        num = cols.get("numeric_features", []) or []
        if not num:
            raise ConfigValidationError(["Для M 4.14 требуется непустой columns.numeric_features"])
        # оставляем только реально существующие
        miss = [c for c in num if c not in df.columns]
        if miss:
            raise KeyError(f"В DataFrame нет колонок из numeric_features: {miss}")

        target_col = cols.get("target_column")
        sample_col = cols.get("sample_column")
        if target_col not in df.columns or sample_col not in df.columns:
            raise KeyError("В данных отсутствуют target_column или sample_column из config")

        df_tr, df_te = self._split_frames(df, sample_col)
        if len(df_tr) == 0:
            raise ValueError("Пустая train-выборка (sample='train').")

        X_tr, y_tr = self._prep_xy(df_tr, num, target_col)
        # 2) два леса с разными random_state
        m1 = self._make_model(self.random_state1)
        m2 = self._make_model(self.random_state2)

        m1.fit(X_tr, y_tr)
        m2.fit(X_tr, y_tr)

        # 3) важности и топ-K списки
        def _importances(model, cols):
            if not hasattr(model, "feature_importances_"):
                raise ValueError("Модель не поддерживает feature_importances_. Задайте другой model_type/модель.")
            return pd.Series(model.feature_importances_, index=cols)

        imp1 = _importances(m1, X_tr.columns).sort_values(ascending=False)
        imp2 = _importances(m2, X_tr.columns).sort_values(ascending=False)

        top_k = min(self.top_k, len(imp1), len(imp2))
        set1 = set(imp1.head(top_k).index)
        set2 = set(imp2.head(top_k).index)
        overlap = len(set1 & set2)
        jaccard = overlap / (2 * top_k - overlap) if (2 * top_k - overlap) > 0 else 1.0

        # 4) агрегированная таблица
        merged = pd.DataFrame({
            "Feature": X_tr.columns
        }).set_index("Feature")
        merged["Imp_1"] = imp1
        merged["Imp_2"] = imp2
        merged["Imp_mean"] = merged[["Imp_1", "Imp_2"]].mean(axis=1)
        merged.sort_values("Imp_mean", ascending=False, inplace=True)

        selected = merged.head(top_k).reset_index().rename(columns={"index": "Feature"})

        # 5) светофор по стабильности
        if jaccard >= 0.70:
            signal = "green"
        elif jaccard >= 0.50:
            signal = "yellow"
        else:
            signal = "red"
        self.test_signal = signal

        # 6) визуал: рассеяние важностей двух лесов
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(merged["Imp_1"], merged["Imp_2"], s=20, alpha=0.6)
        # подсветим выбранные
        sel_mask = merged.index.isin(selected["Feature"])
        ax.scatter(merged.loc[sel_mask, "Imp_1"], merged.loc[sel_mask, "Imp_2"], s=40, alpha=0.9)
        ax.set_xlabel("Importance (Forest #1)")
        ax.set_ylabel("Importance (Forest #2)")
        ax.set_title(f"Two-Forest Importances (top_k={top_k})")
        ax.grid(True, linestyle="--", linewidth=0.5)
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        img_scatter = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 7) HTML
        html = f"""
        <h4>Two-Forest Feature Selection</h4>
        <p><b>Signal:</b> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        <p><b>top_k:</b> {top_k} |
           <b>overlap:</b> {overlap} |
           <b>Jaccard:</b> {jaccard:.3f}</p>
        <h5>Selected features (by mean importance)</h5>
        {selected.to_html(index=False, float_format="%.6f")}
        <h5>Importance scatter (Forest#1 vs Forest#2)</h5>
        <img src="data:image/png;base64,{img_scatter}">
        """
        return {"report": html}
