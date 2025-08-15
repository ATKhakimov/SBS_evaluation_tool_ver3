from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
import pandas as pd

TaskType = Literal["classification", "regression"]

# Специализированное исключение, которое аккумулирует все найденные проблемы
class ConfigValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("\n".join(errors))
        self.errors = errors

@dataclass
class ColumnsConfig:
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    score_column: Optional[str] = None
    prediction_column: Optional[str] = None  # требуется только для classification
    date_column: Optional[str] = None
    target_column: Optional[str] = None
    sample_column: Optional[str] = None
    id_column: Optional[str] = None

@dataclass
class SBSConfig:
    task: TaskType
    columns: ColumnsConfig

def _ensure_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if not isinstance(v, str):
                raise ConfigValidationError([f"`{field_name}` должен быть списком строк; найден элемент типа {type(v)}"])
            out.append(v.strip())
        return out
    raise ConfigValidationError([f"`{field_name}` должен быть списком строк или строкой"])

def _non_empty_str(value: Any, field_name: str, required: bool = True) -> Optional[str]:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        if required:
            raise ConfigValidationError([f"Отсутствует или пустое поле `{field_name}`"])
        return None
    if not isinstance(value, str):
        raise ConfigValidationError([f"`{field_name}` должен быть строкой"])
    return value.strip()

def validate_config(raw_config: Dict[str, Any], df: pd.DataFrame) -> SBSConfig:
    """
    Полная валидация и нормализация конфига относительно переданного df.
    Возвращает нормализованный SBSConfig или бросает ConfigValidationError с детальным списком ошибок.
    """
    errors: List[str] = []

    # -------- базовый уровень --------
    task = raw_config.get("task", None)
    if task not in ("classification", "regression"):
        errors.append("`task` должен быть 'classification' или 'regression'")

    columns = raw_config.get("columns", None)
    if not isinstance(columns, dict):
        errors.append("`columns` обязателен и должен быть объектом (mapping)")

    if errors:
        raise ConfigValidationError(errors)

    # -------- разбор columns --------
    try:
        numeric_features = _ensure_list(columns.get("numeric_features"), "columns.numeric_features")
        categorical_features = _ensure_list(columns.get("categorical_features"), "columns.categorical_features")

        score_column = _non_empty_str(columns.get("score_column"), "columns.score_column", required=False)
        prediction_column = _non_empty_str(columns.get("prediction_column"), "columns.prediction_column", required=False)
        date_column = _non_empty_str(columns.get("date_column"), "columns.date_column", required=True)
        target_column = _non_empty_str(columns.get("target_column"), "columns.target_column", required=True)
        sample_column = _non_empty_str(columns.get("sample_column"), "columns.sample_column", required=True)
        id_column = _non_empty_str(columns.get("id_column"), "columns.id_column", required=True)
    except ConfigValidationError as e:
        # агрегируем внутрь общей ошибки
        errors.extend(e.errors)

    # -------- правила по task --------
    if task == "classification":
        # для классификации должен быть prediction_column (вероятность события)
        if not prediction_column:
            errors.append("Для `task=classification` требуется `columns.prediction_column` (вероятность положительного класса)")
        # score_column опционален (может совпадать с prediction_column), но если задан — должен отличаться по смыслу
    elif task == "regression":
        # для регрессии prediction_column не требуется; можно использовать score_column как выход модели
        prediction_column = prediction_column or None

    # -------- контроль наличия колонок в df --------
    required_columns = {
        c for c in [date_column, target_column, sample_column, id_column, score_column, prediction_column]
        if c is not None
    }
    missing_in_df = [c for c in required_columns if c not in df.columns]
    if missing_in_df:
        errors.append(f"В DataFrame отсутствуют необходимые колонки: {missing_in_df}")

    # Проверяем наличие всех фичей в df
    all_features = list(dict.fromkeys([*numeric_features, *categorical_features]))  # preserve order, remove dups
    missing_features = [c for c in all_features if c not in df.columns]
    if missing_features:
        errors.append(f"Некоторые фичи отсутствуют в данных: {missing_features}")

    # -------- логическая согласованность --------
    # Пересечения между наборами фичей не допускаются
    overlap = sorted(set(numeric_features).intersection(categorical_features))
    if overlap:
        errors.append(f"Фичи не должны дублироваться между numeric и categorical: {overlap}")

    # Пустые списки фичей допустимы, но предупредим
    if not numeric_features and not categorical_features:
        errors.append("Не заданы `numeric_features` и `categorical_features` — список фичей пуст")

    # Предупреждение/ошибка по совпадению служебных колонок с фичами
    reserved = {c for c in [score_column, prediction_column, date_column, target_column, sample_column, id_column] if c}
    bad_reserved_overlap = sorted(reserved.intersection(all_features))
    if bad_reserved_overlap:
        errors.append(f"Служебные колонки не должны входить в список фичей: {bad_reserved_overlap}")

    # Типы данных для базовых полей (мягкая проверка)
    # date_column — желательно datetime/числовая ось времени
    if date_column in df.columns:
        # не приводим принудительно, а только проверяем возможность
        try:
            pd.to_datetime(df[date_column], errors="raise")
        except Exception:
            errors.append(f"`{date_column}` не приводится к datetime; проверь формат дат")

    # target для classification должен быть бинарным (мягкая проверка)
    if task == "classification" and target_column in df.columns:
        unique_vals = pd.Series(df[target_column].dropna().unique())
        if unique_vals.nunique() > 10:
            errors.append(f"`{target_column}` выглядит не бинарным (найдено >10 уникальных значений). "
                          "Проверьте кодировку таргета.")
    # prediction/score должны быть числовыми, если заданы
    for name, col in [("prediction_column", prediction_column), ("score_column", score_column)]:
        if col and col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"`{name}` ('{col}') должен быть числовым")

    if errors:
        raise ConfigValidationError(errors)

    # -------- нормализованный объект --------
    norm_cols = ColumnsConfig(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        score_column=score_column,
        prediction_column=prediction_column,
        date_column=date_column,
        target_column=target_column,
        sample_column=sample_column,
        id_column=id_column,
    )
    return SBSConfig(task=task, columns=norm_cols)
