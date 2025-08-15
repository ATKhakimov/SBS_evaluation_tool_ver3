# core import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_test import BaseModelTest
# default imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
# stat imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.stats import norm
from scipy import sparse
from scipy.special import expit 
# module imports
import itertools
from typing import List, Dict, Union

from joblib import Parallel, delayed, parallel_backend
import numba as nb

# NEW
from core.config_validation import validate_config, ConfigValidationError  # валидатор из предыдущего шага

def _split_frames_from_cfg(df: pd.DataFrame, cfg):
    sc = cfg.columns.sample_column
    if sc is None or sc not in df.columns:
        raise ValueError("В конфиге указан sample_column, но колонки нет в df")
    s = df[sc].astype(str).str.lower()
    df_train = df[s.eq('train')]
    df_test  = df[s.eq('test')]
    df_genpop = df
    return df_train, df_test, df_genpop

def _features_from_cfg(df: pd.DataFrame, cfg):
    num = cfg.columns.numeric_features or []
    cat = cfg.columns.categorical_features or []
    # ВНИМАНИЕ: M2.2/M2.5 требуют численные признаки.
    # По умолчанию берём только numeric_features.
    return list(dict.fromkeys(num)), cat

def _score_from_cfg(cfg):
    # Для классификации используем вероятности события (prediction_column).
    # Для регрессии Gini не определён — бросаем понятную ошибку.
    if cfg.task == "classification":
        if not cfg.columns.prediction_column:
            raise ConfigValidationError(["Для classification требуется columns.prediction_column"])
        return cfg.columns.prediction_column
    raise ConfigValidationError(["Тест M2.* рассчитан на бинарную классификацию (Gini/ROC). Для regression он неприменим."])


class M21_GiniTest(BaseModelTest): 
    """
    Класс для теста M 2.1: Gini Test.

    Проверяет общую способность модели к ранжированию, вычисляя коэффициент Gini 
    для обучающей и тестовой выборки, строит ROC-кривые и визуализирует результаты.
    Выдаёт итоговый сигнал (зелёный, жёлтый или красный) на основе порогов качества.

    Наследует:
        BaseModelTest: Базовый класс для всех модельных тестов.

    Атрибуты:
        * test_signal (str): Итоговый цветовой сигнал ("green", "yellow", "red"),
          рассчитывается на основе значений Gini на train и test.

    Методы:
        * compute_gini(score, target): Вычисляет коэффициент Gini.
        * compute_gini_html(score, target, label): Строит ROC-кривую, возвращает HTML с Gini и графиком.
        * compute_signal(gini_train, gini_test): Определяет сигнал по значениям Gini.
        * run(score_train, target_train, score_test, target_test): Основной метод, запускающий тест.
    """ 
    def __init__(self):
        super().__init__("M 2.1", "Gini Test")

    def compute_gini(self, score: pd.Series, target: pd.Series) -> float:
        """
        Вычисляет коэффициент Gini на основе AUC (Gini = 2 * AUC - 1).

        Параметры:
            * score (pd.Series): Series с предсказанными значениями модели.
            * target (pd.Series): Series с фактическими бинарными метками (0/1).

        Возвращает:
            * float: Значение коэффициента Gini.
        """
        auc = roc_auc_score(target, score)
        return 2 * auc - 1

    def compute_gini_html(self, score: pd.Series, target: pd.Series, label: str) -> str:
        """
        Строит ROC-кривую и вычисляет Gini/AUC, возвращает HTML-блок с графиком.

        Параметры:
            * score (pd.Series): Series с предсказанными значениями модели.
            * target (pd.Series): Series с фактическими бинарными метками (0/1).
            * label (str): Метка для подписи графика (например, "Train" или "Test").

        Возвращает:
            * str: HTML-строка с коэффициентом Gini и изображением ROC-кривой.

        Исключения:
            * ValueError: Если столбец не является числовым и перевод в числовой тип невозможен.
            * ValueError: Если столбец содержит пропущенные значения.
        """
        # Проверяем, что данные можно перевести в числовой формат
        for col_name, col_series in zip(['score', 'target'], [score, target]):
                    if not pd.api.types.is_numeric_dtype(col_series):
                        try:
                            col_series = pd.to_numeric(col_series, errors='raise')
                        except ValueError:
                            raise ValueError(f"Столбец {col_name} не является числовым. Перевод в числовой тип невозможен.")
                    if col_series.isnull().any():
                        raise ValueError(f"Столбец {col_name} содержит пропущенные значения.")
                    
        auc = roc_auc_score(target, score)
        gini = 2 * auc - 1

        fpr, tpr, _ = roc_curve(target, score)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"{label} ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {label}")
        ax.legend()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        encoded = base64.b64encode(buf.getvalue()).decode()

        return gini, f"""
        <h4>{label} Gini</h4>
        <p><b>Gini:</b> {gini:.4f}</p>
        <img src="data:image/png;base64,{encoded}">
        """

    def compute_signal(self, gini_train: float, gini_test: float) -> str:
        """
        Определяет итоговый сигнал на основе значений Gini:
            - "red": если Gini < 0.2 на train или test;
            - "yellow": если Gini < 0.3 хотя бы в одной выборке;
            - "green": в остальных случаях.

        Параметры:
            * gini_train (float): Значение Gini на обучающей выборке.
            * gini_test (float): Значение Gini на тестовой выборке.

        Возвращает:
            * str: Сигнал в формате строки ("red", "yellow", "green").
        """
        if gini_train < 0.2 or gini_test < 0.2:
            return "red"
        elif gini_train < 0.3 or gini_test < 0.3:
            return "yellow"
        return "green"

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, _ = _split_frames_from_cfg(df, cfg)
        target_col = cfg.columns.target_column
        score_col  = _score_from_cfg(cfg)  # см. helper выше

        # Проверки типов/NaN такие же, как внутри compute_gini_html
        gini_train, html_train = self.compute_gini_html(df_train[score_col], df_train[target_col], "Train")
        gini_test,  html_test  = self.compute_gini_html(df_test[score_col],  df_test[target_col],  "Test")
        self.test_signal = self.compute_signal(gini_train, gini_test)

        signal_block = f"""
        <p><b>Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        """
        return {"train": html_train, "test": html_test, "DASHBOARD": signal_block}


@nb.njit(parallel=False, fastmath=True)
def gininumba(y_true: np.array, y_pred: np.array) -> float:
    """
    Рассчитывает коэффициент gini через кривую Лоренца.
    """
    # Сортируем предсказания по убыванию вероятностей и получаем индексы сортировки
    order = np.argsort(-y_pred)    
    # Применяем сортировку к истинным значениям на основе индексов             
    y_sorted = y_true[order]

    total = y_sorted.sum()
    n = y_sorted.size

    if total == 0: 
        return 0 
    
    # Вычисляем накопленные суммы истинных значений (например, убытков) и экспозиции
    cum_y = np.cumsum(y_sorted)
    cum_pop = np.arange(1, n + 1)
    # Вычисляем gini через кривую Лоренца для предсказанных значений
    gini = (2.0 * np.trapz(cum_y / total, cum_pop / n) - 1.0)

    return gini

def gini_one_feature(y_tr: np.array, y_te: np.array, x_tr: np.array, x_te: np.array, name:str):
        """
        Вычисляет индекс Джини для указанной переменной на тренировочном и тестовом наборах данных.
        """
        # Считаем gini на трейн и переводим в %
        g_tr = gininumba(y_tr, x_tr) * 100.0
        # Считаем gini на тесте и переводим в %
        g_te = gininumba(y_te, x_te) * 100.0
        return name, g_tr, g_te, abs(g_tr - g_te)

def gini_for_features_fast(df_tr: pd.DataFrame, df_te: pd.DataFrame, target: str, features: list, n_jobs=-1) -> list:
    """
    Вычисляет индекс Джини для списка факторов на тренировочном и тестовом наборах данных.

    Параметры
    ----------
    * df_tr : pd.DataFrame
          - Тренировочный набор данных.
    * df_te : pd.DataFrame
          - Тестовый набор данных.
    * target : str
          - Имя целевой переменной.
    * features : list
          - Список имен функций, для которых нужно вычислить индекс Джини.
    * n_jobs : int, optional
          - Количество заданий для параллельной обработки. По умолчанию: -1 (используются все доступные ядра).

    Возвращает
    -------
    * list: Список, содержащий результаты вычисления индекса Джини для каждой переменной.
   """
    # Получаем таргет для трайн и тест
    y_tr = df_tr[target].to_numpy(dtype=np.float32)
    y_te = df_te[target].to_numpy(dtype=np.float32)

    with parallel_backend("threading"):
        # Запускаем обработку признаков 
        out = Parallel(n_jobs=n_jobs)(
            delayed(gini_one_feature)(
                y_tr, y_te,
                df_tr[f].to_numpy(dtype=np.float32), # Нам хватит даже для весов, тк точность до 7 знаков
                df_te[f].to_numpy(dtype=np.float32),
                f
            ) for f in features
        )
    return out

# Переопределим M22_GiniFactorsTest с улучшениями после сброса
class M22_GiniFactorsTest(BaseModelTest):
    """
    Класс для теста M 2.2: Gini by Factors.

    Проверяет информативность признаков на обучающей и тестовой выборках с помощью коэффициента Джини.
    Строит графики, расставляет флаги (зеленый/жёлтый/красный) по уровням значимости признаков 
    и определяет общий сигнал качества. Используется для оценки стабильности и значимости отдельных факторов.

    Общая логика:
        1. Для каждого признака считаются значения Gini по train и test (на основе кривой Лоренца).
        2. Вычисляется разница между Gini на train и test.
        3. Каждому признаку присваивается флаг: 
            - "green" — Gini > 5%,
            - "yellow" — 0% < Gini ≤ 5%,
            - "red" — Gini ≤ 0%.
        4. Выводится сигнал:
            - "red" — если есть хотя бы один красный флаг,
            - "yellow" — если более 20% признаков жёлтые,
            - "green" — в остальных случаях.
        5. Генерируется HTML-отчет с графиками и таблицами.

    Атрибуты:
        * test_signal (str): Итоговый сигнал для всего теста ("green", "yellow", "red").
        * individual_features_gini (pd.DataFrame): Таблица со значениями Gini и флагами по каждому признаку.

    Методы:
        * assign_flag(gini): Возвращает флаг на основе значения Gini.
        * compute_signal(flags): Определяет итоговый сигнал по списку флагов.
        * plot_gini_bar(df, column, title): Строит горизонтальную гистограмму Gini для признаков.
        * run(df_train, df_test, target_column, features, n_jobs): Запускает тест и возвращает HTML-отчет.
    """
    def __init__(self):
        super().__init__("M 2.2", "Gini by Factors")
        self.individual_features_gini = pd.DataFrame()

    def assign_flag(self, gini):
        """
        Присваивает цветовой флаг на основе значения Gini:
            - "red": Gini ≤ 0
            - "yellow": Gini ≤ 5
            - "green": Gini > 5

        Параметры:
            * gini (float): Значение Gini (%).

        Возвращает:
            * str: Цвет флага.
        """
        if gini <= 0:
            return "red"
        elif gini <= 5:
            return "yellow"
        else:
            return "green"

    def compute_signal(self, flags: List[str]) -> str:
        """
        Рассчитывает итоговый сигнал по списку флагов:
            - "red": если есть хотя бы один красный.
            - "yellow": если жёлтых более 20%.
            - "green": в остальных случаях.

        Параметры:
            * flags (List[str]): Список цветовых флагов по признакам.

        Возвращает:
            * str: Цвет итогового сигнала ("red", "yellow", "green").
        """
        if "red" in flags:
            return "red"
        yellow_count = flags.count("yellow")
        total = len(flags)
        if yellow_count / total > 0.2:
            return "yellow"
        return "green"

    def plot_gini_bar(self, gini_df: pd.DataFrame, column: str, title: str) -> str:
        """
        Строит горизонтальную гистограмму значений Gini по признакам и кодирует её в base64.

        Параметры:
            * gini_df (pd.DataFrame): Таблица с Gini и флагами.
            * column (str): Название колонки для построения ("Gini_Train" или "Gini_Test").
            * title (str): Заголовок графика.

        Возвращает:
            * str: Base64-кодированное изображение графика.
        """
        # 0.25 inch на признак → 400 фич = 100 inch ≈ 254 см; ограничим
        height = min(0.2 * len(gini_df), 60)
        fig, ax = plt.subplots(figsize=(10, height))
        
        bars = ax.barh(gini_df["Feature"], gini_df[column], color="lightgray")
        for i, flag in enumerate(gini_df[f"{column}_flag"]):
            bars[i].set_color(flag)

        ax.set_xlabel("Gini %")
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle='--', linewidth=0.5)
        
        # Уменьшаем шрифт, чтобы длинные имена влезли
        ax.tick_params(labelsize=8)
        
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def run(self, df: pd.DataFrame, config: dict, n_jobs=3) -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, _ = _split_frames_from_cfg(df, cfg)
        target_column = cfg.columns.target_column
        features, _cat = _features_from_cfg(df, cfg)  # только численные фичи

        if not features:
            raise ConfigValidationError(["Для M 2.2 требуется непустой список columns.numeric_features"])

        # Ниже — твоя исходная проверка наличия/типов/NaN и ускоренная функция gini_for_features_fast
        features = list(features)
        if any(list(map(lambda x: x not in df_train.columns or x not in df_test.columns, features + [target_column]))):
            raise ValueError("Some of the provided columns are missing in the datasets.")
        
        for col_name in features+[target_column]:
            if not pd.api.types.is_numeric_dtype(df_train[col_name]):
                try:
                    df_train[col_name] = pd.to_numeric(df_train[col_name], errors='raise')
                except ValueError:
                    raise ValueError(f"Column {col_name} is not numeric.")
            if not pd.api.types.is_numeric_dtype(df_test[col_name]):
                try:
                    df_test[col_name] = pd.to_numeric(df_test[col_name], errors='raise')
                except ValueError:
                    raise ValueError(f"Column {col_name} is not numeric.")
            if df_train[col_name].isnull().any():
                raise ValueError(f"Column {col_name} contains missing values.")
            if df_test[col_name].isnull().any():
                raise ValueError(f"Column {col_name} contains missing values.")

        records = gini_for_features_fast(df_tr=df_train , df_te=df_test,
                                        target=target_column, features=features,
                                        n_jobs=n_jobs)
        self.individual_features_gini = pd.DataFrame(records, columns=["Feature", "Gini_Train", "Gini_Test", "Delta"]).sort_values(by='Gini_Train')
        self.individual_features_gini["Gini_Train_flag"] = self.individual_features_gini["Gini_Train"].apply(self.assign_flag)
        self.individual_features_gini["Gini_Test_flag"] = self.individual_features_gini["Gini_Test"].apply(self.assign_flag)

        img_train = self.plot_gini_bar(self.individual_features_gini.sort_values(by='Gini_Train'), "Gini_Train", "Gini by Factor (Train)")
        img_test = self.plot_gini_bar(self.individual_features_gini.sort_values(by='Gini_Test'), "Gini_Test", "Gini by Factor (Test)")

        def count_flags(flag_series):
            return {
                "green": (flag_series == "green").sum(),
                "yellow": (flag_series == "yellow").sum(),
                "red": (flag_series == "red").sum()
            }

        train_flags = count_flags(self.individual_features_gini["Gini_Train_flag"])
        test_flags = count_flags(self.individual_features_gini["Gini_Test_flag"])

        signal_train = self.compute_signal(self.individual_features_gini["Gini_Train_flag"].tolist())
        signal_test = self.compute_signal(self.individual_features_gini["Gini_Test_flag"].tolist())

        self.test_signal = "red" if "red" in [signal_train, signal_test] else ("yellow" if "yellow" in [signal_train, signal_test] else "green")

        signal_html = f"""
        <p><b>Signal (Train):</b> <span style='color:{signal_train}; font-weight:bold'>{signal_train.upper()}</span></p>
        <p><b>Signal (Test):</b> <span style='color:{signal_test}; font-weight:bold'>{signal_test.upper()}</span></p>
        <p><b>Overall Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        """

        summary_table = f"""
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>Dataset</th><th style='color:green'>Green</th><th style='color:orange'>Yellow</th><th style='color:red'>Red</th></tr>
            <tr><td>Train</td><td style='color:green'><b>{train_flags["green"]}</b></td><td style='color:orange'><b>{train_flags["yellow"]}</b></td><td style='color:red'><b>{train_flags["red"]}</b></td></tr>
            <tr><td>Test</td><td style='color:green'><b>{test_flags["green"]}</b></td><td style='color:orange'><b>{test_flags["yellow"]}</b></td><td style='color:red'><b>{test_flags["red"]}</b></td></tr>
        </table>
        """

        full_html = f"""
        <h4>Gini by Factors</h4>
        {signal_html}
        {summary_table}
        <br>
        <h5>Gini on Train</h5>
        <img src="data:image/png;base64,{img_train}">
        <h5>Gini on Test</h5>
        <img src="data:image/png;base64,{img_test}">
        <br>
        <h5>Gini Table</h5>
        {self.individual_features_gini.sort_values(by='Gini_Train').to_html(index=False, float_format="%.2f")}
        """
        return {"DASHBOARD": full_html}

# Реализация тестов M 2.4 Gini Dynamics и M 2.5 Gini Uplift
def determine_optimal_freq(dates: pd.Series, max_bins: int = 40) -> str:
    """
    Функция принимает на вход Series с датами и максимальное количество бинов.
    Определяет оптимальную частоту для группировки дат.

    Параметры:
        * dates (pd.Series): Series с датами.
        * max_bins (int): Максимальное количество бинов.

    Возвращает:
        * str: Оптимальная частота для группировки дат.
    """
    dates = pd.to_datetime(dates)
    total_days = (dates.max() - dates.min()).days
    if total_days <= 180:
        return "W"  # weekly
    elif total_days <= 365:
        return "M"  # monthly
    elif total_days <= 3 * 365:
        return "Q"  # quarterly
    else:
        return "A"  # annually

class M24_GiniDynamicsTest(BaseModelTest):
    """
    Класс для теста M 2.4: Gini Dynamics.

    Анализирует динамику коэффициента Gini по времени на обучающей и тестовой выборках.
    Цель — выявить деградацию модели или нестабильность качества прогноза во времени.

    Общая логика:
        1. Данные агрегируются по датам (по умолчанию — помесячно).
        2. Для каждого временного бакета рассчитывается:
            - Gini (%),
            - Количество наблюдений и событий,
            - Среднее значение целевой переменной,
            - Доверительные интервалы (95% и 99%).
        3. Строится график: динамика Gini + количество наблюдений.
        4. Выводится сигнал:
            - "red": если ≥50% временных бакетов имеют Gini < 35%,
            - "yellow": если ≥30% бакетов имеют Gini < 35%,
            - "green": в остальных случаях.

    Атрибуты:
        * test_signal (str): Итоговый цветовой сигнал ("green", "yellow", "red").

    Методы:
        * compute_confidence_interval(gini, n, alpha): Рассчитывает доверительный интервал Gini.
        * compute_table(df, date_column, score_column, target_column): Строит таблицу метрик по времени.
        * determine_signal(gini_values): Вычисляет итоговый сигнал по значению Gini.
        * plot_dynamics(df, title): Строит график динамики Gini и количества наблюдений.
        * run(df_train, df_test, date_column, score_column, target_column): Запускает тест и возвращает HTML-отчёт.
    """
    def __init__(self):
        super().__init__("M 2.4", "Gini Dynamics")

    def compute_confidence_interval(self, gini, n, alpha=0.05):
        """
        Вычисляет доверительный интервал для коэффициента Gini при заданном уровне значимости.

        Параметры:
            * gini (float): Значение коэффициента Gini.
            * n (int): Количество наблюдений.
            * alpha (float): Уровень значимости (по умолчанию 0.05 — 95% интервал).

        Возвращает:
            * Tuple[float, float]: Нижняя и верхняя границы доверительного интервала.
        """
        se = np.sqrt((gini * (1 - gini)) / n) if n > 0 else 0
        z = norm.ppf(1 - alpha / 2)
        return gini - z * se, gini + z * se

    def compute_table(self, df: pd.DataFrame, date_column: str, score_column: str, target_column: str) -> pd.DataFrame:
        """
        Группирует данные по времени и рассчитывает Gini, доверительные интервалы и прочие статистики.

        Параметры:
            * df (pd.DataFrame): Датафрейм с исходными данными.
            * date_column (str): Имя колонки с датами.
            * score_column (str): Имя колонки с прогнозами модели.
            * target_column (str): Имя колонки с фактическими метками.

        Возвращает:
            * pd.DataFrame: Таблица с динамикой Gini и сопутствующей статистикой.
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        freq = "M" #determine_optimal_freq(df.index)
        grouped = df.groupby(pd.Grouper(freq=freq))

        stats = []
        for date, group in grouped:
            if len(group) == 0:
                continue
            n = len(group)
            y_true = group[target_column]
            y_score = group[score_column]
            if len(np.unique(y_true)) < 2:
                gini = 0
            else:
                auc = roc_auc_score(y_true, y_score)
                gini = 2 * auc - 1
            gini_95_low, gini_95_high = self.compute_confidence_interval(gini, n, alpha=0.05)
            gini_99_low, gini_99_high = self.compute_confidence_interval(gini, n, alpha=0.01)
            stats.append((date.to_period(freq).start_time.strftime("%d-%m-%Y"), n, y_true.sum(), y_true.mean(), gini * 100,
                          gini_95_low * 100, gini_95_high * 100,
                          gini_99_low * 100, gini_99_high * 100))

        return pd.DataFrame(stats, columns=[
            "Date", "Observations", "Events", "Mean Target", "Gini %",
            "CI 95% Low", "CI 95% High", "CI 99% Low", "CI 99% High"
        ])

    def determine_signal(self, gini_values: List[float]) -> str:
        """
        Определяет сигнал на основе доли временных бакетов с Gini < 35%.

        Параметры:
            * gini_values (List[float]): Список значений Gini по времени.

        Возвращает:
            * str: Цвет итогового сигнала ("red", "yellow", "green").
        """
        below_35 = sum(1 for g in gini_values if g < 35)
        total = len(gini_values)
        if total == 0:
            return "green"
        ratio = below_35 / total
        if ratio >= 0.5:
            return "red"
        elif ratio >= 0.3:
            return "yellow"
        else:
            return "green"

    def plot_dynamics(self, df: pd.DataFrame, title: str) -> str:
        """
        Строит график динамики Gini и количества наблюдений по времени.

        Параметры:
            * df (pd.DataFrame): Таблица с результатами из compute_table.
            * title (str): Заголовок графика.

        Возвращает:
            * str: base64-кодированное изображение графика.
        """
        fig, ax1 = plt.subplots(figsize=(12, 4))  # чуть шире
        x = pd.to_datetime(df["Date"])

        ax1.plot(x, df["Gini %"], marker="o", label="Gini %", color="green")
        ax1.set_ylabel("Gini %", color="green")

        ax2 = ax1.twinx()
        ax2.plot(x, df["Observations"], marker="x", linestyle="--",
                label="Observations", color="orange")
        ax2.set_ylabel("Observations", color="orange")

        # --- НОВОЕ: форматирование оси X ---
        ax1.xaxis.set_major_locator(mdates.YearLocator())           # крупный тик — раз в год
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))  # мелкие тики — раз в 3 мес
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax1.tick_params(axis='x', which='major', pad=15)             # отступ для подписи года
        ax1.tick_params(axis='x', which='minor', rotation=45)

        fig.autofmt_xdate()   # аккуратно повернуть подписи
        ax1.set_title(title)
        fig.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def run(self, df: pd.DataFrame, config: dict) -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, _ = _split_frames_from_cfg(df, cfg)
        date_column = cfg.columns.date_column
        target_column = cfg.columns.target_column
        score_column  = _score_from_cfg(cfg)  # prediction_column для классификации

        # Прежние проверки типов/NaN
        for column in [score_column, target_column]:
            if not pd.api.types.is_numeric_dtype(df_train[column]) or not pd.api.types.is_numeric_dtype(df_test[column]):
                try:
                    df_train[column] = pd.to_numeric(df_train[column], errors='raise')
                    df_test[column] = pd.to_numeric(df_test[column], errors='raise')
                except ValueError:
                    raise ValueError(f"Столбец {column} не является числовым. Перевод в числовой тип невозможен.")
            if df_train[column].isnull().any() or df_test[column].isnull().any():
                raise ValueError(f"Столбец {column} содержит пропущенные значения.")

        table_train = self.compute_table(df_train, date_column, score_column, target_column)
        table_test  = self.compute_table(df_test,  date_column, score_column, target_column)

        signal_train = self.determine_signal(table_train["Gini %"].tolist())
        signal_test  = self.determine_signal(table_test["Gini %"].tolist())

        self.test_signal = "red" if "red" in [signal_train, signal_test] else (
            "yellow" if "yellow" in [signal_train, signal_test] else "green"
        )

        plot_train = self.plot_dynamics(table_train, "Gini Dynamics - Train")
        plot_test  = self.plot_dynamics(table_test,  "Gini Dynamics - Test")

        signal_html = f"""
        <p><b>Signal (Train):</b> <span style='color:{signal_train}; font-weight:bold'>{signal_train.upper()}</span></p>
        <p><b>Signal (Test):</b> <span style='color:{signal_test}; font-weight:bold'>{signal_test.upper()}</span></p>
        <p><b>Overall Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        """

        html = f"""
        <h4>Gini Dynamics</h4>
        {signal_html}
        <h5>Train</h5>
        <img src="data:image/png;base64,{plot_train}">
        {table_train.to_html(index=False, float_format="%.2f")}
        <br>
        <h5>Test</h5>
        <img src="data:image/png;base64,{plot_test}">
        {table_test.to_html(index=False, float_format="%.2f")}
        """
        return {"DASHBOARD": html}

class M25_GiniUpliftTest(BaseModelTest):
    """
    Класс для теста M 2.5: Gini Uplift.

    Оценивает инкрементальный прирост Gini при пошаговом добавлении признаков в логистическую регрессию.
    Используется для определения важности факторов в контексте всей модели.

    Общая логика:
        1. Вычисляется индивидуальный Gini каждого признака (или используется из M22).
        2. Строится логистическая модель с признаками, добавляемыми по одному в заданном порядке.
        3. На каждом шаге оценивается прирост Gini на train относительно предыдущего шага.
        4. Признаку присваивается флаг:
            - "green" — если прирост > 0.1%,
            - "yellow" — если прирост > 0,
            - "red" — если прирост ≤ 0.
        5. Выводится общий сигнал:
            - "red" — если есть хотя бы один "red" или ≥ 40% признаков "yellow",
            - "yellow" — если ≥ 20% признаков "yellow",
            - "green" — в остальных случаях.
        6. Формируется диаграмма и таблица результатов.

    Атрибуты:
        * test_signal (str): Итоговый сигнал ("green", "yellow", "red").

    Методы:
        * compute_uplift_flag(uplift_val): Присваивает флаг признаку по значению прироста Gini.
        * _plot_uplift_bar(df): Строит график прироста Gini (столбики + линии).
        * run(df_train, df_test, target_column, feature_order, n_jobs, m22_test_object): Запускает тест, возвращает HTML-отчёт.
    """
    def __init__(self):
        super().__init__("M 2.5", "Gini Uplift")

    def compute_uplift_flag(self, uplift_val:float) -> str:
        """
        Присваивает цветовой флаг в зависимости от величины прироста Gini:
            - "green": прирост > 0.1%
            - "yellow": 0% < прирост ≤ 0.1%
            - "red": прирост ≤ 0%

        Параметры:
            * uplift_val (float): Прирост Gini.

        Возвращает:
            * str: Цвет флага.
        """
        if uplift_val > 0.001:
            return "green"
        elif uplift_val > 0:
            return "yellow"
        else:
            return 'red'
         
    def _plot_uplift_bar(self, df: pd.DataFrame) -> str:
        """
        Строит вертикальный график прироста Gini по признакам:
            • Столбики: Gini uplift по каждому признаку, окрашены по флагам.
            • Линии: Gini на train/test по накопленной модели, Gini одиночного признака.

        Параметры:
            * df (pd.DataFrame): Таблица с рассчитанными приростами Gini.

        Возвращает:
            * str: Base64-строка с изображением графика.
        """
        n = len(df)
        # ширина холста растёт, но ограничена 24" (≈60 см), чтобы не упасть браузером
        fig_w = min(1 + 0.25 * n, 24)
        fig, ax = plt.subplots(figsize=(fig_w, 5))

        # цвета с прозрачностью 0.4
        flag2rgba = {
            "green":  (0.2, 0.8, 0.2, 0.4),
            "yellow": (1.0, 0.8, 0.0, 0.4),
            "red":    (1.0, 0.2, 0.2, 0.4),
        }
        bar_colors = [flag2rgba[c] for c in df["Uplift Signal"]]

        x = np.arange(n)
        bars = ax.bar(x, df["Gini Uplift"], color=bar_colors)

        # линии/точки – компактнее
        ax.plot(x, df["Model Train Gini"], color="darkgreen",
                marker="o", markersize=3, linewidth=0.8, label="Gini Train")
        ax.plot(x, df["Model Test Gini"], color="green",
                linestyle="--", marker="x", markersize=3,
                linewidth=0.8, label="Gini Test")
        ax.scatter(x, df["Single Gini"], color="black",
                s=10, label="Single Factor Gini")

        # --- оси / сетка / подписи ------------------------------------------
        # Укорачиваем слишком длинные названия (оставляем первые 25 символов)
        short_labels = [
            (f[:22] + "…") if len(f) > 25 else f
            for f in df["Feature"]
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=7)

        ax.set_ylabel("Gini %")
        ax.set_title("Incremental Gini Uplift by Feature")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        plt.tight_layout()

        # ----- экспорт --------------------------------------------------------
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def run(self, df: pd.DataFrame, config: dict, n_jobs=-1, m22_test_object: "M22_GiniFactorsTest" = None) -> Dict[str, str]:
        cfg = validate_config(config, df)
        df_train, df_test, _ = _split_frames_from_cfg(df, cfg)
        target_column = cfg.columns.target_column
        # Порядок фичей: берём numeric_features (можно позже добавить секцию config['m2']['uplift_feature_order'])
        feature_order, _cat = _features_from_cfg(df, cfg)
        if not feature_order:
            raise ConfigValidationError(["Для M 2.5 требуется непустой список columns.numeric_features"])

        # Проверки типов/NaN и получение individual gini — как у тебя
        for column in feature_order + [target_column]:
            if not pd.api.types.is_numeric_dtype(df_train[column]) or not pd.api.types.is_numeric_dtype(df_test[column]):
                try:
                    df_train[column] = pd.to_numeric(df_train[column], errors='raise')
                    df_test[column] = pd.to_numeric(df_test[column], errors='raise')
                except ValueError:
                    raise ValueError(f"Column {column} must be numeric.")
        if df_train[target_column].isnull().any() or df_test[target_column].isnull().any():
            raise ValueError("Target contains missing values.")

        if m22_test_object and not m22_test_object.individual_features_gini.empty:
            self.individual_features_gini = m22_test_object.individual_features_gini
        else: 
            records = gini_for_features_fast(df_tr=df_train , df_te=df_test,
                                            target=target_column, features=feature_order,
                                            n_jobs=n_jobs)
            self.individual_features_gini = pd.DataFrame(records, columns=["Feature", "Gini_Train", "Gini_Test", "Delta"])
    

        # Готовим данные 
        y_tr = df_train[target_column].to_numpy(np.float32)
        y_ts = df_test[target_column].to_numpy(np.float32)

        X_tr = df_train[feature_order].to_numpy(np.float32)
        X_ts = df_test[feature_order].to_numpy(np.float32)

        # Создаем модель и обучаем ее 
        model = LogisticRegression(solver='saga',
                                   #penalty='l2',
                                   #C=1.0,
                                   max_iter=1000,
                                   n_jobs=n_jobs)
        
        model.fit(X_tr, y_tr)

        # Получаем веса модели
        beta0 = np.float32(model.intercept_[0])
        betas = model.coef_[0].astype(np.float32)

        # Создаем переменные для обновления в цикле
        logits_tr = np.full(len(X_tr), beta0, dtype=np.float32)
        logits_ts = np.full(len(X_ts), beta0, dtype=np.float32)

        records = []

        for i, feat in enumerate(feature_order):
            # Обновляем логиты 
            logits_tr += X_tr[:, i] * betas[i]
            logits_ts += X_ts[:, i] * betas[i]
            # Получаем придикты 
            pred_tr = expit(logits_tr)
            pred_ts = expit(logits_ts)
            # Считаем gini
            g_tr = gininumba(y_tr, pred_tr)
            g_ts = gininumba(y_ts, pred_ts)

            uplift = g_tr - records[i - 1][2] / 100 if i > 0 else g_tr
            uplift_flag = self.compute_uplift_flag(uplift)
            # Gini of single feature
            g_single = self.individual_features_gini[self.individual_features_gini["Feature"] == feat]["Gini_Train"].iloc[0] / 100

            records.append((feat, g_ts * 100, g_tr * 100, g_single * 100, uplift * 100, uplift_flag))

        df_result = pd.DataFrame(records, columns=["Feature", "Model Test Gini", 'Model Train Gini', "Single Gini", "Gini Uplift", 'Uplift Signal'])

        # График прироста
        encoded_plot = self._plot_uplift_bar(df_result)
        
        # Устанавливаем сигнал 
        n_red_fact = len(df_result[df_result['Uplift Signal'] == 'red'])
        n_yellow_fact = len(df_result[df_result['Uplift Signal'] == 'yellow'])
        if n_red_fact > 0 or n_yellow_fact >= len(df_result)*0.4:
            self.test_signal = "red"
        elif n_yellow_fact >= len(df_result)*0.2:
            self.test_signal = "yellow"
        else:
            self.test_signal = "green"

        html = f"""
        <h4>Gini Uplift</h4>
        <p><b>Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        <img src="data:image/png;base64,{encoded_plot}">
        <h5>Details</h5>
        {df_result.to_html(index=False, float_format="%.2f")}
        """

        return {"combined": html}