from dotenv import load_dotenv
from starlette.responses import JSONResponse

load_dotenv()

from fastapi import FastAPI, Response, HTTPException

from google.api_core.exceptions import GoogleAPIError

from app.schemas import WordRequest, WordResponse, HistoryItem
from app.graph import graph

from datetime import datetime
app = FastAPI(
    title="Synonyms & Antonyms API",
    description="Сервис для получения синонимов и антонимов к слову с помощью ИИ",
    version="2.0.0",
)

history: list[HistoryItem] = []

@app.exception_handler(GoogleAPIError)
async def gemini_error_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": f"Ошибка GeminiAPI {exc.message}"})

@app.post("/words", response_model=WordResponse, status_code=200)
async def get_words(request: WordRequest, response: Response):
    """
    Получить 10 синонимов и 10 антонимов к указанному слову в одном запросе.

    - **word**: слово, для которого нужно найти синонимы и антонимы
    """

    result = await graph.ainvoke(
        {"word": request.word, "synonyms": [], "antonyms": [], "result": None}
    )
    if len(result["antonyms"]) < 10 or len(result["synonyms"]) < 10:
        raise HTTPException(status_code=422, detail="LLM вернул меньше 10 слов")
    else:
        history.append(HistoryItem(original_word=request.word, synonyms=result["synonyms"], antonyms=result["antonyms"], timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return result["result"]



@app.get("/history", response_model=list[HistoryItem])
async def get_words_last():
    return history[-10:]