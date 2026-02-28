from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.schemas import WordRequest, WordResponse
from app.graph import graph

app = FastAPI(
    title="Synonyms & Antonyms API",
    description="Сервис для получения синонимов и антонимов к слову с помощью ИИ",
    version="2.0.0",
)


@app.post("/words", response_model=WordResponse)
async def get_words(request: WordRequest):
    """
    Получить 10 синонимов и 10 антонимов к указанному слову в одном запросе.

    - **word**: слово, для которого нужно найти синонимы и антонимы
    """
    result = await graph.ainvoke(
        {"word": request.word, "synonyms": [], "antonyms": [], "result": None}
    )
    return result["result"]
