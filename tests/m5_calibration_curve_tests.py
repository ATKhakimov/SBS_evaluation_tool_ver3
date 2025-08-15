# Core import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_test import BaseModelTest

#Math
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
import statsmodels.api as sm
from io import BytesIO
import base64

# Data
import pandas as pd
import numpy as np

# default imports
import matplotlib.pyplot as plt
import base64

# Settings
from typing import Optional, List, Dict

def binning_data_for_binom(y_true, y_pred):
    # для входных данных формирует новый столбец для биннирования
    pass



def calculate_binom_test(n: int, obs_rate: float, pred_rate: float) -> float:
    """
    Вычисляет метрику симметрии хвостов биномиального распределения
    для оценки соответствия между фактической и предсказанной долей успехов
    в рамках одного бакета.

    Формула:
        result = |P(X ≤ k) - P(X ≥ k)|,
        где X ~ Bin(n, pred_rate),
              k = round(n * obs_rate)

    Интерпретация результата:
        - result ≈ 0  — фактическая доля хорошо согласуется с прогнозом.
        - result ≈ 1  — наблюдаемая доля лежит в одном из хвостов распределения, предсказание плохо согласуется с фактом.

    Параметры:
        * n (int): Общее количество наблюдений в бакете.
        * obs_rate (float): Фактическая доля успехов (mean_true).
        * pred_rate (float): Предсказанная вероятность успеха (mean_pred).

    Возвращает:
        * float: метрика калибровочной симметрии.
    """
    k = np.round(obs_rate * n, 0).astype(int)
    cdf = binom.cdf(k, n, pred_rate)  # Левый хвост        
    pmf = binom.pmf(k, n, pred_rate)  # P(X = k)        
    return abs(cdf - (1 - cdf + pmf)) # Разность хвостов


class M51_TargetRateTest(BaseModelTest): 
    def __init__(self, test_num='M 5.1', test_name='Predicted vs Observed Rate of Target'):
        super().__init__(test_num, test_num)

    def bootstrap_ci(self, y_pred, stat_function=np.mean, n_bootstraps=100, alpha=0.05): #уменьшил до 100
        """
        Вычисляет доверительный интервал с использованием метода бутстрепа с использованием джекнайф-оценок для коррекции асимметрии.

        Параметры:
            * y_pred (array-like): Массив предсказанных значений.
            * stat_function (callable): Функция, используемая для вычисления статистики. По умолчанию - среднее значение.
            * n_bootstraps (int): Количество бутстреп-выборок. По умолчанию - 1000.
            * alpha (float): Уровень значимости. По умолчанию - 0.05.

        Возвращает:
            * tuple: Доверительный интервал в виде кортежа (нижняя граница, верхняя граница).
        """
        y_pred = np.asarray(y_pred)
        theta_hat = stat_function(y_pred)
        n = len(y_pred)
        # Получаем случайные выборки с заменой
        boot_sample = np.random.choice(y_pred, size=(n_bootstraps, n), replace=True)
        boot_stats = np.apply_along_axis(stat_function, axis=1, arr=boot_sample)
        # Оцениваем смещение и асимметрию
        prob_less = np.mean(boot_stats < theta_hat)
        z0 = norm.ppf(prob_less)

        # Джекнайф-оценки
        jack_vals = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(y_pred, i)
            jack_vals[i] = stat_function(jack_sample)
        jack_mean = np.mean(jack_vals)
        a = np.sum((jack_mean - jack_vals)**3) / (6 * np.sum((jack_mean - jack_vals)**2)**1.5)

        # Коррекция квантилей 
        alpha_points = []
        for alpha_i in [alpha/2, 1-alpha/2]:
            z_alpha = norm.ppf(alpha_i)
            z = z0 + (z0 + z_alpha) / (1 - a*(z0 + z_alpha))
            alpha_points.append(norm.cdf(z))

        # Вычисление доверительного интервала
        lower = np.quantile(boot_stats, min(alpha_points))
        upper = np.quantile(boot_stats, max(alpha_points))
        return lower, upper

    def determine_signal(self, mean_true, ci_95: tuple, ci_99: tuple) -> str:
        """
        Определяет сигнал на основе среднего значения и доверительных интервалов.

        Параметры:
            * mean_true (float): Среднее значение.
            * ci_95 (tuple): Доверительный интервал 95%.
            * ci_99 (tuple): Доверительный интервал 99%.

        Возвращает:
            * str: Сигнал ("green", "orange" или "red").
        """
        if ci_95[0] <= mean_true <= ci_95[1]:
            return "green"
        elif ci_99[0] <= mean_true <= ci_99[1]:
            return "orange"
        return "red"

    def plot_target_rate(self, mean_true, mean_pred, ci_95, ci_99) -> str:
        """
        Создает график, сравнивающий наблюдаемый и предсказанный показатели c доверительным интервалом.

        Параметры:
            * mean_true (float): Наблюдаемый показатель.
            * mean_pred (float): Предсказанный показатель.
            * ci_99 (tuple): Доверительный интервал 99%.
            * ci_95 (tuple): Доверительный интервал 95%.

        Возвращает:
            * str: Кодированный в base64 изображение графика.
        """
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(mean_true, color="black", label="Observed Target Rate")
        ax.axhline(mean_pred, color="blue", linestyle="--", label="Predicted Mean")
        ax.fill_between([0, 1], ci_99[0], ci_99[1], color="orange", alpha=0.3, label="99% CI")
        ax.fill_between([0, 1], ci_95[0], ci_95[1], color="green", alpha=0.3, label="95% CI")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(ci_99[1], mean_true) + 0.05)
        ax.set_title("Observed vs Predicted Target Rate")
        ax.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def run(self, score: pd.Series, target: pd.Series) -> Dict[str, str]:
        """
        Выполняет тест для сравнения предсказанного и наблюдаемого показателей, возвращая HTML-код с результатами.

        Этот метод вычисляет средние значения для наблюдаемых и предсказанных данных, определяет доверительные интервалы
        для предсказанных значений с уровнями значимости 95% и 99%, определяет светофор на основе среднего значения и
        доверительных интервалов, создает график для визуализации результатов и формирует HTML-код с результатами теста.


        Параметры:
            * score (pd.Series): Серия предсказанных значений.
            * target (pd.Series): Серия наблюдаемых значений.

        Возвращает:
            * Dict[str, str]: Словарь с HTML-кодом графика и результатами анализа.

        Исключения:
            * ValueError: Если Score или Target не являются числовыми или содержат NaN.
        """
        if not pd.api.types.is_numeric_dtype(score) or not pd.api.types.is_numeric_dtype(target):
            raise ValueError("Score и Target должны быть числовыми")
        if score.isnull().any() or target.isnull().any():
            raise ValueError("Score и Target не должны содержать NaN")

        mean_true = np.mean(target)
        mean_pred = np.mean(score)
        ci_95 = self.bootstrap_ci(score, alpha=0.05)
        ci_99 = self.bootstrap_ci(score, alpha=0.01)
        signal = self.determine_signal(mean_true, ci_95, ci_99)
        self.test_signal = signal

        plot_encoded = self.plot_target_rate(mean_true, mean_pred, ci_95, ci_99)

        html = f"""
        <h4>Predicted vs Observed Rate of Target</h4>
        <p><b>Observed TR:</b> {mean_true:.4f}</p>
        <p><b>Predicted TR:</b> {mean_pred:.4f}</p>
        <p><b>95% CI:</b> ({ci_95[0]:.4f}, {ci_95[1]:.4f})</p>
        <p><b>99% CI:</b> ({ci_99[0]:.4f}, {ci_99[1]:.4f})</p>
        <p><b>Signal:</b> <span style='color:{signal}; font-weight:bold'>{signal.upper()}</span></p>
        <img src="data:image/png;base64,{plot_encoded}">
        """
        return {"combined": html}


class M52_CalibrationCurveByPredictBins(BaseModelTest):
    def __init__(self, test_num='M 5.2', test_name='Calibration Curve (Grouped by Predict Bins)'):
        self.test_signal = None
        super().__init__(test_num, test_name)
    
 
    def determine_signal(self, binom_test_result:pd.Series) -> tuple:
        """
        Определяет сигнал на основе результатов биномиального теста.

        Функция анализирует количество значений в столбце, превышающих пороговые значения 0.99 и 0.95,
        и на основе этого определяет сигнал: "red", "orange" или "green".

        Параметры:
            * binom_test_result (pd.Series): Столбец с результатами биномиального теста.

        Возвращает:
            * tuple: Кортеж, содержащий сигнал (str) и количество значений, превышающих пороги 0.99 и 0.95 (int, int).

        Логика определения сигнала:
            - "red": если более 10% значений превышают 0.99 или более 30% значений превышают 0.95.
            - "orange": если более 10% значений превышают 0.99 или более 10% значений превышают 0.95.
            - "green": в остальных случаях.
        """
        
        n_99 = (binom_test_result > 0.99).sum()
        n_95 = (binom_test_result > 0.95).sum()


        if n_99 > int(len(binom_test_result) * 0.1) and n_95 > int(len(binom_test_result) * 0.3): 
            signal = "red"
        elif n_99 > int(len(binom_test_result) * 0.1) or n_95 > int(len(binom_test_result) * 0.1):
            signal = "orange"
        else:
            signal = "green"

        
        return signal, n_99, n_95


    def plot_calibration_curve(self, grouped: pd.DataFrame) -> str:
        """
        Строит кривую калибровки для заданного DataFrame. Функция строит график, отображающий кривую среднего предсказания и точки со средним таргетом.
        Подписи осей и заголовок графика настроены для лучшей читаемости.

        Параметры:
        ----------
        * grouped : pd.DataFrame
            DataFrame, содержащий столбцы 'bin', 'mean_true' и 'mean_pred', 'count'.
            - 'bin': интервалы предсказанных вероятностей.
            - 'mean_true': среднее значение истинного таргета в каждом интервале.
            - 'mean_pred': среднее значение предсказанного таргета в каждом интервале.
            - 'count': количество наблюдений в каждом интервале.

        Возвращает:
        -------
        * str
            График в формате base64, который можно встроить в HTML.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the observed mean target
        ax.plot(grouped["bin"].astype(str), grouped["mean_true"], label="Observed", marker="o", linestyle='-', color='blue')

        # Plot the predicted mean score
        ax.plot(grouped["bin"].astype(str), grouped["mean_pred"], label="Predicted", linestyle="--", color="gray")

        # Add labels and title
        ax.set_xlabel("Predicted probability intervals")
        ax.set_ylabel("Mean rate")
        ax.set_title("Calibration Curve Test")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add legend
        ax.legend()

        # Create a secondary y-axis for the count of observations
        ax2 = ax.twinx()
        ax2.bar(grouped["bin"].astype(str), grouped["count"], alpha=0.3, color='green', label="Count")
        ax2.set_ylabel("Count of Observations")

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode()


    def run(self, score: pd.Series, target: pd.Series, num_groups:int = 20, bins_step: float = None) -> dict:
        """
        Выполняет тест калибровочной кривой, группируя предсказания по интервалам и оценивая соответствие между фактическими и предсказанными значениями.

        Метод разбивает предсказания на интервалы, вычисляет средние фактические и предсказанные значения для каждого интервала,
        определяет сигнал на основе биномиального теста и строит график калибровочной кривой.

        Параметры:
            * score (pd.Series): Серия с предсказанными значениями.
            * target (pd.Series): Серия с фактическими значениями.
            * num_groups (int, optional): Количество групп для разбиения предсказаний. По умолчанию 20.
            * bins_step (float, optional): Шаг для разбиения предсказаний на интервалы. По умолчанию None.

        Возвращает:
            * dict: Словарь, содержащий HTML-код с результатами теста, включая график калибровочной кривой, сигнал и статистику по группам.

        Логика выполнения:
            1. Разбивает предсказания на интервалы с использованием заданного шага или количества групп.
            2. Вычисляет средние фактические и предсказанные значения для каждого интервала.
            3. Применяет биномиальный тест для оценки соответствия между фактическими и предсказанными значениями.
            4. Определяет сигнал на основе результатов биномиального теста.
            5. Формирует HTML-код с результатами теста, включая график, сигнал и статистику по группам.
        """
        if not pd.api.types.is_numeric_dtype(score) or not pd.api.types.is_numeric_dtype(target):
            raise ValueError("Score и Target должны быть числовыми.")
        if score.isnull().any() or target.isnull().any():
            raise ValueError("Score и Target не должны содержать NaN.")
        if not bins_step and len(score) < num_groups:
            raise ValueError("Недостаточно данных для указанного числа групп. Используйте bins_step или уменьшете кол-во групп.")
        if len(score) != len(target):
            raise ValueError("Длина Score и Target должна быть одинаковой.")

        df = pd.DataFrame({"score": score, "target": target})

        # Bin the predictions
        if bins_step:
            bins = np.arange(0, score.max() + bins_step, bins_step)
            df["bin"] = pd.cut(df["score"], bins=bins, include_lowest=True)
        else: 
            df['bin'] = pd.qcut(df['score'], q=num_groups)

        grouped = df.groupby("bin").agg(
            count=("target", "count"),
            mean_true=("target", "mean"),
            mean_pred=("score", "mean")
        ).reset_index()

        grouped['bin_test_result'] = grouped.apply(lambda row: calculate_binom_test(row['count'], row['mean_true'], row['mean_pred']), axis=1)


        # Determine signal
        self.test_signal, n_99, n_95 = self.determine_signal(grouped['bin_test_result'])

        # Plot
        img_encoded = self.plot_calibration_curve(grouped)

        html = f"""
        <h4>Calibration Curve by Bins</h4>
        <p><b>Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        <img src='data:image/png;base64,{img_encoded}'>
        <br>
        <h5>Confidence Interval Exceedances</h5>
        <table border="1">
            <tr>
                <th>Yellow</th>
                <th>Red</th>
            </tr>
            <tr>
                <td>Beyond 95% CI: >{int(len(grouped) * 0.1)}</td>
                <td>Beyond 95% CI: >{int(len(grouped) * 0.3)}</td>
            </tr>
            <tr>
                <td>Beyond 99% CI: >{int(len(grouped) * 0.1)}</td>
                <td>Beyond 99% CI: >{int(len(grouped) * 0.1)}</td>
            </tr>
        </table>
        <p><b>Actual number of exceedances beyond 95% CI:</b> {n_95}</p>
        <p><b>Actual number of exceedances beyond 99% CI:</b> {n_99}</p>
        <p><b>Traffic Light Test:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal}</span></p>
        <br>
        <h5>Group Stats</h5>
        {grouped.to_html(index=False, float_format="%.4f")}
        """
        return {"combined": html}


class M53_CalibrationCurveByDates(BaseModelTest):
    def __init__(self, test_num='M 5.3', test_name='Calibration Curve (Grouped by Dates)'):
        self.test_signal = None
        super().__init__(test_num, test_name)

    def determine_signal(self, binom_test_result:pd.Series) -> tuple:
        """
        Определяет сигнал на основе результатов биномиального теста.

        Функция анализирует количество значений в столбце, превышающих пороговые значения 0.99 и 0.95,
        и на основе этого определяет сигнал: "red", "orange" или "green".

        Параметры:
            * binom_test_result (pd.Series): Столбец с результатами биномиального теста.

        Возвращает:
            * tuple: Кортеж, содержащий сигнал (str) и количество значений, превышающих пороги 0.99 и 0.95 (int, int).

        Логика определения сигнала:
            - "red": если более 10% значений превышают 0.99 или более 30% значений превышают 0.95.
            - "orange": если более 10% значений превышают 0.99 или более 10% значений превышают 0.95.
            - "green": в остальных случаях.
        """
        
        n_99 = (binom_test_result > 0.99).sum()
        n_95 = (binom_test_result > 0.95).sum()


        if n_99 > int(len(binom_test_result) * 0.1) or n_95 > int(len(binom_test_result) * 0.3): 
            signal = "red"
        elif n_99 > int(len(binom_test_result) * 0.1) or n_95 > int(len(binom_test_result) * 0.1):
            signal = "orange"
        else:
            signal = "green"

        return signal, n_99, n_95
    
    def plot_calibration_over_time(self, grouped: pd.DataFrame) -> str:
        """
        Строит кривую калибровки для заданного DataFrame по временной колнке. Функция строит график, отображающий кривую среднего предсказания и точки со средним таргетом.
        Подписи осей и заголовок графика настроены для лучшей читаемости.

        Параметры:
        ----------
        * grouped : pd.DataFrame
            DataFrame, содержащий столбцы 'date_str', 'mean_true' и 'mean_pred', 'count'.
            - 'date_str': дата в виде строки в формате "YYYY-MM-DD".
            - 'mean_true': среднее значение истинного таргета в каждом периоде.
            - 'mean_pred': среднее значение предсказанного таргета в каждом периоде.
            - 'count': количество наблюдений в каждом периоде.

        Возвращает:
        -------
        * str
            График в формате base64, который можно встроить в HTML.

        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the observed mean target
        ax.plot(grouped["date_str"], grouped["mean_true"], label="Observed", marker="o", linestyle='-', color='blue')

        # Plot the predicted mean score
        ax.plot(grouped["date_str"], grouped["mean_pred"], label="Predicted", linestyle="--", color="gray")

        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Target Rate")
        ax.set_title("Calibration over Time")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add legend
        ax.legend(loc='upper left')

        # Create a secondary y-axis for the count of observations
        ax2 = ax.twinx()
        ax2.bar(grouped["date_str"], grouped["count"], alpha=0.3, color='green', label="Count")
        ax2.set_ylabel("Count of Observations")

        # Add legend for the secondary y-axis
        ax2.legend(loc='upper right')

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode()


    def run(self, date_column: pd.Series, score_column: pd.Series, target_column: pd.Series, period:str='M') -> dict:
        """
        Выполняет тест калибровочной кривой, группируя данные по временным периодам, и оценивает соответствие между фактическими и предсказанными значениями.

        Метод разбивает данные на временные периоды (недели, месяцы, кварталы, годы), вычисляет средние фактические и предсказанные значения для каждого периода,
        определяет сигнал на основе биномиального теста и строит график калибровочной кривой по времени.

        Параметры:
            * date_column (pd.Series): Серия с датами.
            * score_column (pd.Series): Серия с предсказанными значениями.
            * target_column (pd.Series): Серия с фактическими значениями.
            * period (str, optional): Период группировки данных. Возможные значения:
                - 'W': недельная частота.
                - 'M': месячная частота (по умолчанию).
                - 'Q': квартальная частота.
                - 'Y': годовая частота.

        Возвращает:
            * dict: Словарь, содержащий HTML-код с результатами теста, включая график калибровочной кривой, сигнал и статистику по периодам.

        Пример использования:
            ```python
            model_test = M43_CalibrationCurveByDates()
            result = model_test.run(date_column, score_column, target_column, period='M')
            print(result['combined'])
            ```
        """

        if not pd.api.types.is_numeric_dtype(score_column) or not pd.api.types.is_numeric_dtype(target_column):
            raise ValueError("Score и Target должны быть числовыми.")
        if score_column.isnull().any() or target_column.isnull().any():
            raise ValueError("Score и Target не должны содержать NaN.")
        if len(score_column) != len(target_column) != len(date_column):
            raise ValueError("Длина Score и Target должна быть одинаковой.")            

        # Подготовим данные для анализа
        df = pd.DataFrame({"date": date_column, "score": score_column, "target": target_column})

        try:
            df["date"] = pd.to_datetime(df["date"])
        except:
            raise ValueError("Колонка с датой должна содержать даты и быть в формате даты.")            
        
        
        # Группируем данные с заданным шагом
        grouped = df.groupby(df["date"].dt.to_period(period)).agg(
            count=("target", "count"),
            mean_true=("target", "mean"),
            mean_pred=("score", "mean")
        ).reset_index()
        grouped["date_str"] = grouped["date"].dt.strftime("%Y-%m-%d")

        grouped["result"] = grouped.apply(lambda row: calculate_binom_test(row['count'], row['mean_true'], row['mean_pred']), axis=1)

        # Determine signal
        self.test_signal, n_99, n_95 = self.determine_signal(grouped['result'])

        # Plot
        img_encoded = self.plot_calibration_over_time(grouped=grouped)

        html = f"""
        <h4>Calibration Curve by Date</h4>
        <p><b>Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        <img src='data:image/png;base64,{img_encoded}'>
        <br>
        <h5>Confidence Interval Exceedances</h5>
        <table border="1">
            <tr>
                <th>Yellow</th>
                <th>Red</th>
            </tr>
            <tr>
                <td>Beyond 95% CI: >{int(len(grouped) * 0.1)}</td>
                <td>Beyond 95% CI: >{int(len(grouped) * 0.3)}</td>
            </tr>
            <tr>
                <td>Beyond 99% CI: >{int(len(grouped) * 0.1)}</td>
                <td>Beyond 99% CI: >{int(len(grouped) * 0.1)}</td>
            </tr>
        </table>
        <p><b>Actual number of exceedances beyond 95% CI:</b> {n_95}</p>
        <p><b>Actual number of exceedances beyond 99% CI:</b> {n_99}</p>
        <p><b>Traffic Light Test:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal}</span></p>
        <br>
        <h5>Group Stats</h5>
        {grouped.to_html(index=False, float_format="%.4f")}
        """
        return {"combined": html}


class M44_CalibrationCurveByMobs(BaseModelTest):
    def __init__(self, test_num='M 4.4', test_name='Calibration Curve (Grouped by Mobs)'):
        self.test_signal = None
        super().__init__(test_num, test_name)
   
    def determine_signal(self, binom_test_result:pd.Series) -> tuple:
        """
        Определяет сигнал на основе результатов биномиального теста.

        Функция анализирует количество значений в столбце, превышающих пороговые значения 0.99 и 0.95,
        и на основе этого определяет сигнал: "red", "orange" или "green".

        Параметры:
            * binom_test_result (pd.Series): Столбец с результатами биномиального теста.

        Возвращает:
            * tuple: Кортеж, содержащий сигнал (str) и количество значений, превышающих пороги 0.99 и 0.95 (int, int).

        Логика определения сигнала:
            - "red": если более 10% значений превышают 0.99 или более 30% значений превышают 0.95.
            - "orange": если более 10% значений превышают 0.99 или более 10% значений превышают 0.95.
            - "green": в остальных случаях.
        """
        
        n_99 = (binom_test_result > 0.99).sum()
        n_95 = (binom_test_result > 0.95).sum()


        if n_99 > int(len(binom_test_result) * 0.1) or n_95 > int(len(binom_test_result) * 0.3): 
            signal = "red"
        elif n_99 > int(len(binom_test_result) * 0.1) or n_95 > int(len(binom_test_result) * 0.1):
            signal = "orange"
        else:
            signal = "green"

        return signal, n_99, n_95

    def plot_calibration_over_mob(self, grouped: pd.DataFrame) -> str:
        """
        Строит кривую калибровки для заданного DataFrame по колнке для мобов. Функция строит график, отображающий кривую среднего предсказания и точки со средним таргетом.
        Подписи осей и заголовок графика настроены для лучшей читаемости.

        Параметры:
        ----------
        * grouped : pd.DataFrame
            DataFrame, содержащий столбцы 'mob_group', 'mean_true' и 'mean_pred', 'count'.
            - 'mob_group': значение для группировки данных.
            - 'mean_true': среднее значение истинного таргета в каждом периоде.
            - 'mean_pred': среднее значение предсказанного таргета в каждом периоде.
            - 'count': количество наблюдений в каждом периоде.

        Возвращает:
        -------
        * str
            График в формате base64, который можно встроить в HTML.

        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the observed mean target
        ax.plot(grouped["mob_group"], grouped["mean_true"], label="Observed", marker="o", linestyle='-', color='blue')

        # Plot the predicted mean score
        ax.plot(grouped["mob_group"], grouped["mean_pred"], label="Predicted", linestyle="--", color="gray")

        # Add labels and title
        ax.set_xlabel("Mobs")
        ax.set_ylabel("Target Rate")
        ax.set_title("Calibration over Time")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add legend
        ax.legend(loc='upper left')

        # Create a secondary y-axis for the count of observations
        ax2 = ax.twinx()
        ax2.bar(grouped["mob_group"], grouped["count"], alpha=0.3, color='green', label="Count")
        ax2.set_ylabel("Count of Observations")

        # Add legend for the secondary y-axis
        ax2.legend(loc='upper right')

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode()

    def run(self, score_column: pd.Series, target_column: pd.Series, mob_column: pd.Series) -> dict:
        """
        Выполняет тест калибровочной кривой, группируя данные по значениям мобов, и оценивает соответствие между фактическими и предсказанными значениями.

        Метод группирует данные по уникальным значениям мобов, вычисляет средние фактические и предсказанные значения для каждой группы,
        определяет сигнал на основе биномиального теста и строит график калибровочной кривой по мобам.

        Параметры:
            score_column (pd.Series): Серия с предсказанными значениями.
            target_column (pd.Series): Серия с фактическими значениями.
            mob_column (pd.Series): Серия с идентификаторами мобов.

        Возвращает:
            dict: Словарь, содержащий HTML-код с результатами теста, включая график калибровочной кривой, сигнал и статистику по группам мобов.

        Описание:
            Метод выполняет следующие шаги:
            1. Группирует данные по уникальным значениям мобов.
            2. Вычисляет средние фактические и предсказанные значения для каждой группы мобов.
            3. Применяет биномиальный тест для оценки соответствия между фактическими и предсказанными значениями.
            4. Определяет сигнал на основе результатов биномиального теста.
            5. Формирует HTML-код с результатами теста, включая график, сигнал и статистику по группам мобов.

        Пример использования:
            ```python
            model_test = M44_CalibrationCurveByMobs()
            result = model_test.run(score_column, target_column, mob_column)
            print(result['combined'])
            ```
        """
        if not pd.api.types.is_numeric_dtype(score_column) or not pd.api.types.is_numeric_dtype(target_column):
            raise ValueError("Score и Target должны быть числовыми.")
        if score_column.isnull().any() or target_column.isnull().any():
            raise ValueError("Score и Target не должны содержать NaN.")
        if len(score_column) != len(target_column) != len(mob_column):
            raise ValueError("Длина Score и Target должна быть одинаковой.")            

        # Подготовка данных
        df = pd.DataFrame({"score": score_column, "target": target_column, "mob_group": mob_column.astype(str)})
        
        grouped = df.groupby("mob_group").agg(
            count=("target", "count"),
            mean_true=("target", "mean"),
            mean_pred=("score", "mean")
                                ).reset_index()

        # Расчет биномиального теста
        grouped['result'] = grouped.apply(lambda row: calculate_binom_test(row['count'], row['mean_true'], row['mean_pred']), axis=1)
        # Определение сигнала
        self.test_signal, n_99, n_95 = self.determine_signal(grouped['result'])
        # График кривой калибровки
        img_encoded = self.plot_calibration_over_mob(grouped)
        # Сводная информация
        html = f"""
        <h4>Calibration Curve by Mobs</h4>
        <p><b>Signal:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal.upper()}</span></p>
        <img src='data:image/png;base64,{img_encoded}'>
        <br>
        <h5>Confidence Interval Exceedances</h5>
        <table border="1">
            <tr>
                <th>Yellow</th>
                <th>Red</th>
            </tr>
            <tr>
                <td>Beyond 95% CI: >{int(len(grouped) * 0.1)}</td>
                <td>Beyond 95% CI: >{int(len(grouped) * 0.3)}</td>
            </tr>
            <tr>
                <td>Beyond 99% CI: >{int(len(grouped) * 0.1)}</td>
                <td>Beyond 99% CI: >{int(len(grouped) * 0.1)}</td>
            </tr>
        </table>
        <p><b>Actual number of exceedances beyond 95% CI:</b> {n_95}</p>
        <p><b>Actual number of exceedances beyond 99% CI:</b> {n_99}</p>
        <p><b>Traffic Light Test:</b> <span style='color:{self.test_signal}; font-weight:bold'>{self.test_signal}</span></p>
        <br>
        <h5>Group Stats</h5>
        {grouped.to_html(index=False, float_format="%.4f")}
        """

        return {"combined": html}


# Summary block for group M4
def generate_group4_summary(test_results: dict) -> str:
    rows = []
    for code, result in test_results.items():
        signal = result.get("signal", "unknown")
        description = result.get("description", "")
        color = signal if signal in ["green", "orange", "red"] else "gray"
        rows.append(f"<tr><td>{code}</td><td>{description}</td><td style='color:{color}'><b>{signal.upper()}</b></td></tr>")

    summary = f"""
    <h4>Group M4 Summary</h4>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr><th>Test</th><th>Description</th><th>Signal</th></tr>
        {''.join(rows)}
    </table>
    """
    return summary
