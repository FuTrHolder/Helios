from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from youtube.pipeline import BlogToYouTubePipeline


app = FastAPI(
    title="Helios Blog to YouTube API",
    version="0.1.0",
)


pipeline: BlogToYouTubePipeline | None = None


class GenerateRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
    )

    content: str = Field(
        ...,
        min_length=1,
    )

    url: Optional[str] = None

    max_scenes: int = Field(
        default=1,
        ge=1,
        le=10,
    )

    scene_duration: float = Field(
        default=10.0,
        gt=0,
    )

    num_frames: int = Field(
        default=33,
        ge=33,
    )

    height: int = Field(
        default=384,
        gt=0,
    )

    width: int = Field(
        default=640,
        gt=0,
    )

    steps: int = Field(
        default=2,
        ge=1,
        le=10,
    )

    seed: int = Field(
        default=42,
    )


@app.on_event("startup")
def startup_event():
    global pipeline

    print(
        "[API] Loading Helios pipeline..."
    )

    pipeline = BlogToYouTubePipeline()

    print(
        "[API] Helios pipeline ready."
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
    }


@app.post("/api/generate")
def generate(request: GenerateRequest):

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Helios pipeline is not ready.",
        )

    try:
        result = pipeline.generate(
            title=request.title,
            content=request.content,
            url=request.url,
            max_scenes=request.max_scenes,
            scene_duration=request.scene_duration,
            num_frames=request.num_frames,
            height=request.height,
            width=request.width,
            steps=request.steps,
            seed=request.seed,
        )

        return {
            "status": "completed",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
