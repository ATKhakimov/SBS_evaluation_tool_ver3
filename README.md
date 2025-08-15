
# 📘 Документация по использованию модуля `sbs_evaluation_tool`

`sbs_evaluation_tool` — это модуль для автоматического формирования HTML-отчета по качеству модели, включая PSI, Gini, мультиколлинеарность, стабильность и т.д.

---

##  Структура модуля

```
sbs_evaluation_tool/
├── core/
│   └── base_test.py          # Базовые классы: BaseModelTest, TestGroupBuilder, ModelReportBuilder
├── tests/
│   ├── m0_data_quality.py    # M 0.1 — 0.3
│   ├── m1_distribution_tests.py
│   ├── m2_gini_tests.py
│   ├── m3_multicollinearity.py
│   ├── m4_model_performance.py  # ⭐ M 4.1 — 4.14 (обновлен с новой логикой M4.10)
│   ├── m5_model_stability.py
│   └── psi_calculator.py     # PSICalculator, PSICalculatorExtended
├── demo_m4_tests.ipynb      # 📓 Демонстрационный ноутбук M4 тестов
└── README_sbs_evaluation_tool.md
```

---

##  Базовые классы (`core/base_test.py`)

### `BaseModelTest`

Абстрактный класс для всех тестов. Каждый тест реализует метод:

```python
def run(self, **kwargs) -> Dict[str, str]:
    ...
```

### `TestGroupBuilder`

Группа тестов в одном блоке отчета:

```python
group = TestGroupBuilder("M2", "M 2. Gini Tests")
group.add_test(M21_GiniTest())
group.run_all_tests(test_params={...})
```

### `ModelReportBuilder`

Основной HTML-отчет с улучшенными цветами светофора:

```python
builder = ModelReportBuilder("My Model Report")
builder.add_group(group)
html = builder.generate_html()  # Используется единый светло-желтый цвет для yellow/orange сигналов
```

**Цветовая схема светофоров:**
- 🟢 Зеленый: `#e6f4ea` - отличные результаты
- 🟡 Желтый/Оранжевый: `#fff3cd` - требует внимания (унифицированный цвет)
- 🔴 Красный: `#fbeaea` - критические проблемы

---

## � Миграция с предыдущих версий

### Изменения в M4.10 (Обязательно к прочтению!)

**Старая версия M4.10:**
```python
# Простой запуск без параметров
test = M410_MissingValuesImpactTest()
```

**Новая версия M4.10:**
```python
# Требуется указать time_column для временного анализа
test = M410_MissingValuesImpactTest(
    missing_threshold=0.8,  # Порог пропусков (80%)
    time_column='date',     # ОБЯЗАТЕЛЬНО: колонка с временем
    n_time_buckets=30      # Количество временных бакетов
)
```

**Что изменилось:**
1. **Добавлен обязательный параметр `time_column`** - нужно указать колонку с временными метками
2. **Новая логика анализа** - анализ динамики пропусков во времени вместо простой гистограммы
3. **Новые графики** - график динамики с доверительными интервалами + heatmap
4. **Новая логика светофора** - по количеству вылетов за доверительные интервалы

**Требования к данным:**
- Должна быть колонка с временными метками (datetime или int)
- Рекомендуется >= 100 записей для стабильного анализа
- Временные метки должны покрывать разные периоды

**Пример адаптации кода:**
```python
# Было
tests = [M410_MissingValuesImpactTest()]

# Стало  
tests = [M410_MissingValuesImpactTest(time_column='created_at')]
```

### Другие изменения

**Размеры графиков:**
- Все графики стандартизированы до размера 12x8 дюймов
- Heatmap'ы стали квадратными для лучшей читаемости

**Цветовая схема:**
- Унифицированы цвета предупреждений (yellow/orange) на #fff3cd

**Классы отчетов:**
- `ModelReportBuilder` заменил старые классы генерации отчетов

### Рекомендации по миграции

1. **Проверьте параметры M4.10** - добавьте `time_column`
2. **Обновите размеры графиков** - если переопределяли figsize
3. **Проверьте цвета** - если кастомизировали цветовую схему
4. **Протестируйте на малом датасете** - новый M4.10 может быть медленнее

### Обратная совместимость

- Все остальные тесты работают без изменений
- API базовых классов не изменился
- Форматы отчетов остались прежними

## 📊 Демонстрационный блокнот

**Файл:** `demo_m4_tests.ipynb`

Полный пример использования всех M4 тестов на реальных данных:

**Включает:**
- 🔄 Загрузку и предобработку данных (Titanic, California Housing)
- 🤖 Обучение CatBoost моделей (классификация и регрессия)
- 🧪 Демонстрацию всех 14+ M4 тестов
- 📊 Визуализацию результатов с стандартизированными графиками
- 📋 Генерацию комплексного HTML отчета
- 💡 Рекомендации по использованию в production

**Датасеты в демо:**
- **Titanic** - бинарная классификация выживаемости
- **California Housing** - регрессия стоимости жилья

**Технологии:** CatBoost, scikit-learn, pandas, numpy, matplotlib, seaborn, SHAP

**Запуск:**
```bash
# Откройте ноутбук в VS Code или Jupyter
# Все библиотеки импортируются автоматически
# Пошаговое выполнение с объяснениями
```

---

## Тестовые модули и примеры использования

### 🔹 M 0. Data Quality (`m0_data_quality.py`)

**Назначение:** Проверка качества исходных данных.

- `M01_DataSummaryTest`: анализ количества строк, поиск дубликатов, определение мин/макс дат по выборкам.
- `M02_SampleOverlayTest`: анализ пересечений между выборками по ключу.
- `M03_ObsVsPredTest`: сравнение факта и прогноза по выборкам.

**Пример использования:**
```python
from sbs_evaluation_tool.tests.m0_data_quality import M01_DataSummaryTest
group = TestGroupBuilder("M0", "Data Quality")
group.add_test(M01_DataSummaryTest())
group.run_all_tests(test_params={
    "M 0.1": {"df": df, "date_column": "date", "key_column": "id"}
})
```

---

### 🔹 M 1. Distribution PSI (`m1_distribution_tests.py`)

**Назначение:** Оценка схожести распределений между выборками.

- `M11_TrainVsGenpopTest`: расчет PSI между train и genpop по признакам/таргету.
- `M12_TestVsGenpopTest`: расчет PSI между test и genpop.

**Пример использования:**
```python
from sbs_evaluation_tool.tests.m1_distribution_tests import M11_TrainVsGenpopTest
group = TestGroupBuilder("M1", "Distribution PSI")
group.add_test(M11_TrainVsGenpopTest())
group.run_all_tests(test_params={
    "M 1.1": {"df_train": df_train, "df_genpop": df_genpop, "features": ["score", ...]}
})
```

---

### 🔹 M 2. Gini (`m2_gini_tests.py`)

**Назначение:** Оценка качества модели и признаков по метрике Gini.

- `M21_GiniTest`: расчет Gini по предсказаниям.
- `M22_GiniFactorsTest`: расчет Gini по каждому признаку.
- `M24_GiniDynamicsTest`: динамика Gini во времени.
- `M25_GiniUpliftTest`: прирост Gini при добавлении признаков.

**Пример использования:**
```python
from sbs_evaluation_tool.tests.m2_gini_tests import M22_GiniFactorsTest
group = TestGroupBuilder("M2", "Gini Tests")
group.add_test(M22_GiniFactorsTest())
group.run_all_tests(test_params={
    "M 2.2": {"df": df, "features": ["score", ...], "target_column": "target"}
})
```

---

### 🔹 M 4. Model Performance (`m4_model_performance.py`)

**Назначение:** Комплексный анализ производительности и качества модели для blackbox алгоритмов с визуализацией и светофорными сигналами.

**⚠️ Все графики стандартизированы до размера 12x8 для единообразия отчетов**

**Раздел 4.1. Качество работы алгоритма:**
- `M41_MetricConfidenceIntervalTest`: доверительные интервалы метрик качества с бутстрапом
  - Поддержка классификации и регрессии
  - Настраиваемое количество итераций бутстрапа
  - Графики распределения метрик и scatter plot (для регрессии)
- `M42_SampleSizeAdequacyTest`: проверка достаточности размера обучающей выборки
  - Learning curves с доверительными интервалами
  - Анализ стабилизации качества модели
- `M43_CalibrationTest`: калибровка модели (**только для бинарной классификации**)
  - Reliability diagram с Brier Score
  - Анализ недо/переоценки вероятностей

**Раздел 4.2. Стабильность результатов:**
- `M44_OverfittingTest`: выявление переобучения
  - Сравнение train/test метрик
  - Статистические тесты на значимость различий
- `M45_MetricStabilityTest`: стабильность метрики во времени
  - Временные ряды с доверительными интервалами
  - Анализ трендов и волатильности
- `M46_CategoryQualityTest`: качество модели в разрезе категорий
  - Сравнение метрик по категориям
  - Выявление проблемных сегментов

**Раздел 4.3. Качество отбора признаков:**
- `M47_ShapImportanceTest`: сравнение SHAP importance между train/test ⚠️ *долгое выполнение*
  - SHAP Summary plots для train и test выборок
  - Сравнительный анализ важности признаков
  - Выявление нестабильности feature importance
- `M48_FeatureContributionTest`: анализ адекватности вклада признаков
  - SHAP-based анализ направления влияния признаков
  - Проверка соответствия бизнес-логике
  - Выявление неожиданного поведения модели
- `M49_UpliftTest`: поиск избыточных признаков ⚠️ *долгое выполнение*
  - Feature uplift curves
  - Определение оптимального количества признаков
  - Анализ переобучения на признаках
- `M410_MissingValuesImpactTest`: ⭐ **ОБНОВЛЕН** - анализ пропущенных значений с временной динамикой
  - **Новая логика**: График динамики пропусков по времени с доверительными интервалами
  - **Светофор по вылетам**: Определяется количеством выходов за доверительные интервалы
  - Временные бакеты (до 50) для анализа стабильности пропусков
  - Большая квадратная heatmap пропусков по времени
  - Таблица признаков с высоким уровнем пропусков (>порог)

**Раздел 4.4. Альтернативный отбор факторов:**
- `M411_TargetCorrelationTest`: корреляция признаков с таргетом
  - Horizontal bar charts корреляций
  - Ранжирование по силе связи с таргетом
- `M412_FeatureCorrelationTest`: корреляция признаков между собой
  - **Квадратная heatmap 12x12** корреляционной матрицы
  - Выявление мультиколлинеарности
  - Рекомендации по удалению избыточных признаков
- `M413_VIFTest`: анализ мультиколлинеарности (VIF)
  - Variance Inflation Factor для каждого признака
  - Цветные пороги (VIF=5, VIF=10)
  - Horizontal bar chart VIF значений
- `M414_TwoForestSelectionTest`: двухлесовой отбор признаков ⚠️ *долгое выполнение*
  - Сравнение важности в разных случайных лесах
  - Стабильность feature importance
  - Ранжирование признаков по надежности

**Пример использования M4.10 (обновленный):**
```python
from sbs_evaluation_tool.tests.m4_model_performance import M410_MissingValuesImpactTest

# Новая версия с анализом временной динамики
test = M410_MissingValuesImpactTest(threshold=0.8)  # Только threshold параметр
group = TestGroupBuilder("M4", "Model Performance")
group.add_test(test)
group.run_all_tests(test_params={
    "M 4.10": {
        "X": dataframe_with_features,
        "time_column": "date_column"  # Обязательно для временного анализа
    }
})

# Тест создает:
# 1. График динамики пропусков по времени с доверительными интервалами
# 2. Детальную таблицу проблемных признаков  
# 3. Большую квадратную heatmap пропусков по времени
# 4. Светофорную оценку по количеству вылетов за доверительные интервалы
```

**Пример для классификации:**
```python
from sbs_evaluation_tool.tests.m4_model_performance import M41_MetricConfidenceIntervalTest
from sklearn.ensemble import RandomForestClassifier

# Тест доверительных интервалов для классификации
test = M41_MetricConfidenceIntervalTest(model_type='classification', n_iterations=300)
group = TestGroupBuilder("M4", "Model Performance")
group.add_test(test)
group.run_all_tests(test_params={
    "M 4.1": {"y_true": y_test, "y_pred": y_pred}
})

# Для регрессии
test_reg = M41_MetricConfidenceIntervalTest(model_type='regression')
group.run_all_tests(test_params={
    "M 4.1": {"y_true": y_true, "y_pred": y_pred}
})
```

---

### 🔹 M 5. Model Stability (`m5_model_stability.py`)

**Назначение:** Оценка стабильности модели по PSI.

- `M53_ScorePSIStabilityTest`: PSI по score.
- `M54_FactorsPSIStabilityTest`: PSI по фичам.

**Пример использования:**
```python
from sbs_evaluation_tool.tests.m5_model_stability import M53_ScorePSIStabilityTest
group = TestGroupBuilder("M5", "Model Stability")
group.add_test(M53_ScorePSIStabilityTest())
group.run_all_tests(test_params={
    "M 5.3": {"df_train": df_train, "df_test": df_test, "score_column": "score"}
})
```

---

### 🔹 PSICalculator (`psi_calculator.py`)

**Назначение:** Расчет PSI между выборками.

**Класс:** `PSICalculator`

```python
from sbs_evaluation_tool.tests.psi_calculator import PSICalculator
calc = PSICalculator(bins=10)
result = calc.calculate(expected, actual, target_expected, target_actual)
```
Возвращает словарь с PSI по группам: `all`, `target_0`, `target_1`.

**Вспомогательные функции:**
- `plot_distribution(...)`, `plot_psi(...)` — визуализация распределений и PSI.

---

##  PSICalculator

Файл: `psi_calculator.py`

### Класс: `PSICalculator`

```python
calc = PSICalculator(bins=10)
result = calc.calculate(expected, actual, target_expected, target_actual)
```

Возвращает словарь с PSI по группам:
- `all`
- `target_0`
- `target_1`

### `plot_distribution(...)`, `plot_psi(...)`

Вспомогательные функции для визуализации

---

##  Пример использования

```python
from sbs_evaluation_tool.core.base_test import ModelReportBuilder, TestGroupBuilder
from sbs_evaluation_tool.tests.m2_gini_tests import M22_GiniFactorsTest
from sbs_evaluation_tool.tests.m4_model_performance import M410_MissingValuesImpactTest

df = load_your_data()

# Пример с M2 тестом
group_m2 = TestGroupBuilder("M2", "Gini Tests")
group_m2.add_test(M22_GiniFactorsTest())
group_m2.run_all_tests(test_params={
    "M 2.2": {"df": df, "features": [...], "target_column": "target"}
})

# Пример с обновленным M4.10 тестом
group_m4 = TestGroupBuilder("M4", "Model Performance - Missing Values")
group_m4.add_test(M410_MissingValuesImpactTest(threshold=0.8))
group_m4.run_all_tests(test_params={
    "M 4.10": {"X": df_features, "time_column": "date"}
})

# Создание единого отчета с стандартизированными графиками
builder = ModelReportBuilder("Model Monitoring Report")
builder.add_group(group_m2)
builder.add_group(group_m4)

html = builder.generate_html()
with open("report.html", "w", encoding='utf-8') as f:
    f.write(html)
```

---

## 📦 Зависимости

Для работы модуля требуются следующие библиотеки:

**Основные:**
- `pandas` - работа с данными
- `numpy` - численные вычисления  
- `scikit-learn` - машинное обучение и метрики
- `matplotlib` - построение графиков
- `seaborn` - статистическая визуализация
- `tqdm` - прогресс-бары

**Дополнительные (для M4 тестов):**
- `scipy` - статистические функции
- `statsmodels` - VIF расчеты
- `shap` - интерпретируемость моделей (опционально)

**Установка:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm scipy statsmodels
pip install shap  # опционально для SHAP тестов
```

---

## ⚠️ Предупреждения о производительности

Некоторые тесты могут выполняться длительное время:

**Очень долгие тесты (> 10 минут):**
- `M47_ShapImportanceTest` - расчет SHAP values для больших данных
- `M49_UpliftTest` - множественное переобучение моделей
- `M410_MissingValuesImpactTest` - ⭐ **ОБНОВЛЕН** - временной анализ с доверительными интервалами
- `M414_TwoForestSelectionTest` - обучение множества случайных лесов

**Умеренно долгие тесты (1-10 минут):**
- `M41_MetricConfidenceIntervalTest` - бутстрап с 300+ итераций
- `M42_SampleSizeAdequacyTest` - обучение на разных размерах выборки

**🔄 Изменения в M4.10:**
- **Старая версия**: Простая гистограмма пропущенных значений
- **Новая версия**: Комплексный анализ динамики пропусков во времени
- **Новые параметры**: Требуется `time_column` для временного анализа
- **Новые выходы**: График динамики, доверительные интервалы, heatmap
- **Новая логика светофора**: По количеству вылетов за доверительные интервалы

**Рекомендации:**
- Уменьшите `n_iterations` для ускорения
- Ограничьте количество признаков через `top_features`
- Используйте выборку данных для первичного тестирования
- Для M4.10: убедитесь в наличии `time_column` в данных
- Запускайте долгие тесты в отдельном процессе

---

##  Настройка входных параметров

Каждый тест принимает параметры через `test_params`, например:

```python
"M 2.4": {
    "df": df,
    "date_column": "event_date",
    "score_column": "score",
    "target_column": "target"
}
```

---

##  Как создать свой модуль с тестом

1. **Создайте новый файл в папке `tests/`, например `mX_my_test.py`.**

2. **Определите свой класс теста, унаследованный от `BaseModelTest`:**
```python
from sbs_evaluation_tool.core.base_test import BaseModelTest

class MX_MyCustomTest(BaseModelTest):
    def __init__(self):
        super().__init__(test_code="M X.Y", test_name="My Custom Test")

    def run(self, **kwargs):
        # Реализуйте логику теста
        result = ...
        return {"result": result}
```

3. **Добавьте тест в `__init__.py` соответствующего пакета для автозагрузки:**
```python
from .mX_my_test import MX_MyCustomTest
```

4. **Используйте ваш тест через `TestGroupBuilder`:**
```python
from sbs_evaluation_tool.tests.mX_my_test import MX_MyCustomTest
group = TestGroupBuilder("MX", "My Custom Group")
group.add_test(MX_MyCustomTest())
group.run_all_tests(test_params={
    "M X.Y": {"df": df, ...}
})
```

5. **Добавьте описание теста и его параметров в README для удобства пользователей.**

---

##  Советы

- Убедись, что в `__init__.py` ты импортировал нужные тесты
- Названия тестов в `test_params` должны совпадать с `test_code`
- Используй `group.run_all_tests(test_params=...)` для запуска группы
- Для новых тестов придерживайся структуры: конструктор, метод `run`, описание параметров
- Для визуализации результатов используйте встроенные функции или добавьте свои

---
