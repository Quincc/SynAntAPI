from __future__ import annotations

import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.schemas import LLMWordList, WordItem, WordResponse


class GraphState(TypedDict):
    word: str
    synonyms: list[WordItem]
    antonyms: list[WordItem]
    result: WordResponse | None


SYNONYM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — лингвистический помощник. "
            "Пользователь даёт тебе слово, подбери ровно 10 синонимов к нему. "
            "Верни только список из 10 слов, без пояснений.",
        ),
        ("human", "Слово: {word}"),
    ]
)

ANTONYM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — лингвистический помощник. "
            "Пользователь даёт тебе слово, подбери ровно 10 антонимов к нему. "
            "Верни только список из 10 слов, без пояснений.",
        ),
        ("human", "Слово: {word}"),
    ]
)


def _build_llm() -> ChatGoogleGenerativeAI:
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url:
        os.environ.setdefault("HTTPS_PROXY", proxy_url)
        os.environ.setdefault("HTTP_PROXY", proxy_url)
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.7,
        transport="rest",
    )


def generate_synonyms(state: GraphState) -> GraphState:
    llm = _build_llm()
    chain = SYNONYM_PROMPT | llm.with_structured_output(LLMWordList)
    response: LLMWordList = chain.invoke({"word": state["word"]})
    items = [WordItem(word=w, type="synonym") for w in response.words]
    return {"synonyms": items}


def generate_antonyms(state: GraphState) -> GraphState:
    llm = _build_llm()
    chain = ANTONYM_PROMPT | llm.with_structured_output(LLMWordList)
    response: LLMWordList = chain.invoke({"word": state["word"]})
    items = [WordItem(word=w, type="antonym") for w in response.words]
    return {"antonyms": items}


def format_result(state: GraphState) -> GraphState:
    return {
        "result": WordResponse(
            original_word=state["word"],
            synonyms=state["synonyms"],
            antonyms=state["antonyms"],
        ),
    }


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("generate_synonyms", generate_synonyms)
    builder.add_node("generate_antonyms", generate_antonyms)
    builder.add_node("format_result", format_result)
    builder.add_edge(START, "generate_synonyms")
    builder.add_edge(START, "generate_antonyms")
    builder.add_edge("generate_synonyms", "format_result")
    builder.add_edge("generate_antonyms", "format_result")
    builder.add_edge("format_result", END)
    return builder.compile()


graph = build_graph()
