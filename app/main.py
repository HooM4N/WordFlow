import os
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.schemas import GenerateRequest, SimilarRequest
from app.services import process_run_upload, generate_text_service, get_similar_service

app = FastAPI(title="WordFlow Dashboard")

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    """
    Serves the web dashboard HTML.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/api/load-run")
async def load_run(
    checkpoint: UploadFile = File(...),
    logs: UploadFile = File(...),
    tokenizer: UploadFile = File(...),
    config: UploadFile = File(...)
):
    """
    Endpoint to load model artifacts into memory.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    try:
        temp_dir = os.path.join(os.path.dirname(__file__), ".temp_run")
        return await process_run_upload(checkpoint, logs, tokenizer, config, temp_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load run data: {str(e)}")

@app.post("/api/generate")
def generate(req: GenerateRequest):
    """
    Endpoint for autoregressive story generation.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    try:
        story = generate_text_service(req)
        return {"story": story}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/similar")
def similar(req: SimilarRequest):
    """
    Endpoint for querying similar words via embeddings.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    try:
        results = get_similar_service(req)
        return {"results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
