from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.schemas import WordRequest, WordResponse
from app.graph import graph

app = FastAPI(
    title="Synonyms & Antonyms API",
    description="Сервис для получения синонимов и антонимов к слову с помощью ИИ",
    version="1.0.0",
)


@app.post("/words", response_model=WordResponse)
async def get_words(request: WordRequest):
    """
    Получить 10 синонимов или антонимов к указанному слову.

    - **word**: слово, для которого нужно найти синонимы/антонимы
    - **type**: `synonyms` или `antonyms`
    """
    result = await graph.ainvoke(
        {"word": request.word, "type": request.type, "result": None}
    )
    return result["result"]
