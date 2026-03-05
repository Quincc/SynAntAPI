from dotenv import load_dotenv
from starlette.responses import JSONResponse
from collections import Counter
import json
load_dotenv()

from fastapi import FastAPI, Response, HTTPException

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.db_class import Request, Base, create

from google.api_core.exceptions import GoogleAPIError

from app.schemas import WordRequest, WordResponse, HistoryItem, SynonymsResponse, AntonymsResponse, StatsResponse
from app.graph import graph

from datetime import datetime, timedelta


SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

history_dict: dict[str, HistoryItem] = {}
request_counter: Counter = Counter()
request_timestamps: list[datetime] = []
RATE_LIMIT = 2
RATE_LIMIT_SECONDS = 60

app = FastAPI(
    title="Synonyms & Antonyms API",
    description="Сервис для получения синонимов и антонимов к слову с помощью ИИ",
    version="2.0.0",
)

create(engine)
SessionLocal = sessionmaker(autoflush=False, bind=engine)
db = SessionLocal()

@app.exception_handler(GoogleAPIError)
async def gemini_error_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": f"Ошибка GeminiAPI {exc.message}"})

@app.post("/words", response_model=WordResponse, status_code=200)
async def get_words(request: WordRequest, response: Response):
    """
    Получить 10 синонимов и 10 антонимов к указанному слову в одном запросе.

    Args:
        request (str): Слово для получения синонимов и антонимов

    :return: {word: str, synonyms: list[WordItem], antonyms: list[WordItem], result: WordResponse}

    """

    request_timestamps[:] = [t for t in request_timestamps if datetime.now() - t < timedelta(seconds=RATE_LIMIT_SECONDS)]
    if len(request_timestamps) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, 2 requests per minute")
    request_timestamps.append(datetime.now())

    result = await graph.ainvoke(
        {"word": request.word, "synonyms": [], "antonyms": [], "result": None}
    )
    synonyms = result["synonyms"]
    antonyms = result["antonyms"]
    if len(result["antonyms"]) < 10 or len(result["synonyms"]) < 10:
        raise HTTPException(status_code=422, detail="LLM вернул меньше 10 слов")

    history_dict[request.word] = HistoryItem(original_word=request.word, synonyms=synonyms, antonyms=antonyms, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    request_counter[request.word] += 1
    db.add(Request(original_word=request.word, synonyms=json.dumps([item.model_dump() for item in synonyms]), antonyms=json.dumps([item.model_dump() for item in antonyms])))
    db.commit()
    return result["result"]

@app.get("/words/{word}/synonyms", response_model=SynonymsResponse, status_code=200)
async def get_synonyms(word: str):
    """
    Получить список синонимов к указанному слову.

    - **word**: слово для которого надо получить синонимы
    """
    if word not in history_dict:
        raise HTTPException(status_code=404, detail="Слово не найдено")

    return  SynonymsResponse(original_word=word, synonyms = history_dict[word].synonyms)

@app.get("/words/{word}/antonyms", response_model=AntonymsResponse, status_code=200)
async def get_antonyms(word: str):
    """
    Получить список антонимов к указанному слову.

    - **word**: слово для которого надо получить антонимы
    """
    if word not in history_dict:
        raise HTTPException(status_code=404, detail="Слово не найдено")

    return AntonymsResponse(original_word=word, antonyms = history_dict[word].antonymys)

@app.get("/history", response_model=list[HistoryItem])
async def get_words_last():
    """
    Получить 10 последних запросов.

    :return: list({word: synonyms})
    """
    return list(history_dict.values())[-10::]

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Получить статистику:\n
    **word**: Популярное слово\n
    **count**: Количество запросов

    :return: StatsResponse(popular_word, count_request)
    """
    if not request_counter:
        raise HTTPException(status_code=404, detail="Запросы не найдены")
    word, count = request_counter.most_common(1)[0]
    return StatsResponse(popular_word=word, count_request=count)