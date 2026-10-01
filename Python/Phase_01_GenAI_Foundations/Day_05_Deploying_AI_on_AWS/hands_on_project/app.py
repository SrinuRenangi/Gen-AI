"""
=============================================================================
Project: Enterprise AI Deployment on AWS (Day 05)
File: app.py
Description: Production FastAPI Microservice serving GenAI models on AWS.
             Deployable to AWS App Runner, AWS ECS Fargate, or AWS Lambda.
=============================================================================
"""

import os
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bedrock_client import BedrockAIClient

app = FastAPI(
    title="AWS Enterprise AI Serving Gateway",
    description="High-performance GenAI API service deployed on AWS with Amazon Bedrock, ECS, and Lambda support.",
    version="1.0.0",
)

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Bedrock Client
ai_client = BedrockAIClient()


# ---------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str = Field(..., example="Explain AWS ECS Fargate vs Lambda for deploying AI.")
    system_prompt: Optional[str] = Field(
        default="You are an expert Cloud & AI Solutions Architect.",
        example="You are an expert Cloud & AI Solutions Architect."
    )
    model_id: Optional[str] = Field(
        default=BedrockAIClient.MODEL_CLAUDE_3_5_SONNET,
        example=BedrockAIClient.MODEL_CLAUDE_3_5_SONNET,
    )
    temperature: Optional[float] = Field(default=0.4, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)


class ChatResponse(BaseModel):
    model_id: str
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., example=["Cloud computing", "Generative AI on AWS"])
    model_id: Optional[str] = Field(default=BedrockAIClient.MODEL_TITAN_EMBEDDING)


class EmbeddingResponse(BaseModel):
    model_id: str
    embeddings: List[List[float]]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    region: str
    mode: str
    timestamp: float


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check():
    """
    Health check probe for AWS Application Load Balancers (ALB) and ECS Task definitions.
    """
    return HealthResponse(
        status="healthy",
        region=ai_client.region_name,
        mode="mock" if ai_client.is_mock else "live_aws",
        timestamp=time.time(),
    )


@app.post("/v1/chat/completions", response_model=ChatResponse, tags=["Generative AI"])
async def chat_completion(request: ChatRequest):
    """
    Synchronous generation endpoint calling Amazon Bedrock foundation models.
    """
    start_time = time.perf_counter()
    try:
        result = ai_client.generate_text(
            prompt=request.prompt,
            system_prompt=request.system_prompt or "You are a helpful assistant.",
            model_id=request.model_id or BedrockAIClient.MODEL_CLAUDE_3_5_SONNET,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature or 0.5,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ChatResponse(
            model_id=result["model_id"],
            response_text=result["text"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=round(elapsed_ms, 2),
            status=result["status"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.post("/v1/chat/stream", tags=["Generative AI"])
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time token streaming.
    """
    async def token_generator():
        async for token in ai_client.generate_stream(
            prompt=request.prompt,
            system_prompt=request.system_prompt or "You are a helpful assistant.",
            model_id=request.model_id or BedrockAIClient.MODEL_CLAUDE_3_5_SONNET,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature or 0.5,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["Embeddings"])
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate dense vector embeddings using Amazon Titan for RAG or search.
    """
    start_time = time.perf_counter()
    try:
        vectors = ai_client.generate_embeddings(
            texts=request.texts,
            model_id=request.model_id or BedrockAIClient.MODEL_TITAN_EMBEDDING,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return EmbeddingResponse(
            model_id=request.model_id or BedrockAIClient.MODEL_TITAN_EMBEDDING,
            embeddings=vectors,
            count=len(vectors),
            latency_ms=round(elapsed_ms, 2),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
