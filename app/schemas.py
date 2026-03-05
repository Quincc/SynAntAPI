from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from datetime import datetime


class WordRequest(BaseModel):
    """Запрос от пользователя."""

    word: str = Field(..., description="Слово, для которого нужно найти синонимы и антонимы", min_length=2, max_length=50, pattern= "^[A-Za-zА-Яа-яЁё\-]+$")


class WordItem(BaseModel):
    """Один элемент результата (синоним или антоним)."""

    word: str = Field(..., description="Синоним или антоним")
    type: Literal["synonym", "antonym"] = Field(..., description="Тип: synonym или antonym")


class LLMWordList(BaseModel):
    """Структурированный ответ от LLM — список слов."""

    words: list[str] = Field(..., description="Список из 10 слов")


class WordBaseResponse(BaseModel):
    """Исходное слово"""

    original_word: str = Field(..., description="Исходное слово")


class SynonymsResponse(WordBaseResponse):
    """Ответ сервера со списком синонимов"""

    synonyms: list[WordItem] = Field(..., description="Список из 10 синонимов")


class AntonymsResponse(WordBaseResponse):
    """Ответ сервера со списком синонимов"""

    antonyms: list[WordItem] = Field(..., description="Список из 10 антонимов")


class WordResponse(SynonymsResponse, AntonymsResponse):
    """Ответ сервера со списком синонимов и антонимов."""


class HistoryItem(WordResponse):
    """Ответ сервера со списком синонимов, антонимов и времени"""

    timestamp: datetime = Field(..., description="Время")


class StatsResponse(BaseModel):
    """Ответ сервера со статистикой: слово, количество запросов"""

    popular_word: str = Field(..., description="Самое популярное слово")
    count_request: int = Field(..., description="Количество запросов")