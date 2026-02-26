from __future__ import annotations

import os
from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.schemas import LLMWordList, WordItem, WordResponse


class GraphState(TypedDict):
    word: str
    type: Literal["synonyms", "antonyms"]
    result: WordResponse | None


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — лингвистический помощник. "
            "Пользователь даёт тебе слово и просит подобрать ровно 10 {type} к нему. "
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


def generate(state: GraphState) -> GraphState:
    llm = _build_llm()
    structured_llm = llm.with_structured_output(LLMWordList)
    chain = PROMPT_TEMPLATE | structured_llm

    type_label = "синонимов" if state["type"] == "synonyms" else "антонимов"
    response: LLMWordList = chain.invoke({"word": state["word"], "type": type_label})

    item_type: Literal["synonym", "antonym"] = (
        "synonym" if state["type"] == "synonyms" else "antonym"
    )
    items = [WordItem(word=w, type=item_type) for w in response.words]

    return {
        **state,
        "result": WordResponse(original_word=state["word"], items=items),
    }


def build_graph():
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()


graph = build_graph()
