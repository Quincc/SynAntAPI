from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, TEXT

class Base(DeclarativeBase): pass

class Request(Base):
    """
    Attributes:
        id = Column(Integer): id
        original_word = Column(String): Слово для запроса
        synonyms = Column(TEXT): Список синонимов
        antonyms = Column(TEXT): Список антонимов
    """

    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, autoincrement=True)
    original_word = Column(String)
    synonyms = Column(TEXT)
    antonyms = Column(TEXT)

def create(engine):
    return Base.metadata.create_all(engine)