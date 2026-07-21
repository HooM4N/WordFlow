from pydantic import BaseModel

class GenerateRequest(BaseModel):
    max_tokens: int = 100
    temperature: float = 0.8
    seed: int | None = None

class SimilarRequest(BaseModel):
    word: str
