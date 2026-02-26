from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WordRequest(BaseModel):
    """Запрос от пользователя."""

    word: str = Field(..., description="Слово, для которого нужно найти синонимы или антонимы")
    type: Literal["synonyms", "antonyms"] = Field(
        ..., description="Тип запроса: synonyms (синонимы) или antonyms (антонимы)"
    )


class WordItem(BaseModel):
    """Один элемент результата (синоним или антоним)."""

    word: str = Field(..., description="Синоним или антоним")
    type: Literal["synonym", "antonym"] = Field(..., description="Тип: synonym или antonym")


class WordResponse(BaseModel):
    """Ответ сервера со списком синонимов/антонимов."""

    original_word: str = Field(..., description="Исходное слово")
    items: list[WordItem] = Field(..., description="Список из 10 синонимов или антонимов")


class LLMWordList(BaseModel):
    """Структурированный ответ от LLM — список слов."""

    words: list[str] = Field(..., description="Список из 10 синонимов или антонимов")
