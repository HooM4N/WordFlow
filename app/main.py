from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from app.services import InferenceService

app = FastAPI(title="WordFlow Dashboard")

# Initialize our stateful backend service
inference_service = InferenceService(runs_dir="runs")

# Ensure static directory exists
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- Pydantic Models for Requests ---
class LoadRunRequest(BaseModel):
    run_id: str

class GenerateRequest(BaseModel):
    prompt: str | None = None
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 10
    seed: int | None = None

class SimilarRequest(BaseModel):
    word: str
    top_n: int = 5

# --- Routes ---
@app.get("/")
async def serve_frontend():
    return FileResponse("app/static/index.html")

@app.get("/api/runs")
async def get_runs():
    """Returns metadata and logs for all available runs"""
    return {"runs": inference_service.list_all_runs()}

@app.post("/api/load")
async def load_run(request: LoadRunRequest):
    """Loads a specific run into memory"""
    try:
        stats = inference_service.load_run(request.run_id)
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    """Generates autoregressive text"""
    try:
        text = inference_service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            seed=request.seed
        )
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/similar")
async def get_similar(request: SimilarRequest):
    """Finds cosine similarities in the embedding matrix"""
    try:
        results = inference_service.similar(
            word=request.word,
            top_n=request.top_n
        )
        return {"similar_words": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
