from pydantic import BaseModel

class LoadRunRequest(BaseModel):
    run_id: str

class GenerateRequest(BaseModel):
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 10
    seed: int | None = None

class SimilarRequest(BaseModel):
    word: str
    top_n: int = 5
