import logging
import asyncio
import openai
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import ValidationError
from contextlib import asynccontextmanager

from app.config import settings
from app.llm_client import llm_client
from app.schemas import (
    ChatRequest,
    ChatResponse,
    AssistantAnswer,
    ResearchRequest,
    ResearchResponse,
)
from app.agent import run_research
from app.rag.retriever import retrieve, format_context
from app.rag import ingest as ingest_module
from app.cache import response_cache
from app.middleware import (
    RateLimitMiddleware,
    RequestTrackingMiddleware,
    ErrorHandlingMiddleware,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_ingestion_status = {"running": False, "last_result": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Assistant API starting up...")
    logger.info(f"Provider: {settings.LLM_PROVIDER}, Model: {settings.OPENAI_MODEL if settings.LLM_PROVIDER == 'openai' else settings.VLLM_MODEL}")
    if settings.FALLBACK_PROVIDER:
        logger.info(f"Fallback provider: {settings.FALLBACK_PROVIDER}")
    yield
    logger.info("AI Assistant API shutting down...")


app = FastAPI(
    title="AI Assistant API",
    description="LLM assistant with RAG, tool calling, and structured output.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestTrackingMiddleware)


@app.get("/health")
async def health_check():
    circuit_state = llm_client._circuit_breaker.state.value
    return {
        "status": "ok",
        "provider": settings.LLM_PROVIDER,
        "fallback": settings.FALLBACK_PROVIDER or None,
        "circuit_breaker": circuit_state,
        "cache": response_cache.stats,
        "ingestion": _ingestion_status,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history_dicts = request.get_normalized_history()
        answer = await llm_client.chat(
            request.message,
            history=history_dicts,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        return ChatResponse(answer=answer, sources=[], confidence=1.0)
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Please try again later.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please wait before retrying.")
    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {type(e).__name__}")


@app.post("/chat/tools")
async def chat_with_tools(request: ChatRequest):
    try:
        history_dicts = request.get_normalized_history()
        result = await llm_client.chat_with_tools(
            request.message,
            history=history_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return result
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Please try again later.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please wait before retrying.")
    except Exception as e:
        logger.exception("Tool-calling request failed")
        raise HTTPException(status_code=500, detail=f"Tool-calling request failed: {type(e).__name__}")


@app.post("/chat/rag", response_model=ChatResponse)
async def chat_rag(request: ChatRequest):
    try:
        # Offload synchronous ChromaDB vector search to threadpool to keep event loop unblocked
        chunks = await asyncio.to_thread(retrieve, request.message, top_k=4)
        context = format_context(chunks)

        raw = await llm_client.chat_structured(
            request.message,
            context=context,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        validated = AssistantAnswer(**raw)

        return ChatResponse(
            answer=validated.answer,
            sources=validated.sources,
            confidence=validated.confidence,
        )
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"Model returned invalid structured output: {e}")
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Please try again later.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please wait before retrying.")
    except Exception as e:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail=f"RAG request failed: {type(e).__name__}")


@app.post("/chat/structured", response_model=ChatResponse)
async def chat_structured(request: ChatRequest):
    """Produces structured JSON output guaranteed to follow the AssistantAnswer schema."""
    try:
        raw = await llm_client.chat_structured(
            request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        validated = AssistantAnswer(**raw)
        return ChatResponse(
            answer=validated.answer,
            sources=validated.sources,
            confidence=validated.confidence,
            structured_data=raw,
        )
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"Model returned invalid structured output: {e}")
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Please try again later.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please wait before retrying.")
    except Exception as e:
        logger.exception("Structured chat request failed")
        raise HTTPException(status_code=500, detail=f"Structured chat failed: {type(e).__name__}")


@app.post("/chat/research", response_model=ResearchResponse)
async def chat_research(request: ResearchRequest):
    """Agentic research loop: the model decides its own tool sequence and stop point."""
    try:
        result = await run_research(
            request.message,
            mode=request.mode,
            max_steps=request.max_steps,
            temperature=request.temperature if request.temperature is not None else 0.2,
            top_p=request.top_p if request.top_p is not None else 1.0,
            max_tokens=request.max_tokens if request.max_tokens is not None else 1200,
            inject_failure=request.inject_failure,
        )
        return ResearchResponse(**result)
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Please try again later.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please wait before retrying.")
    except Exception as e:
        logger.exception("Research request failed")
        raise HTTPException(status_code=500, detail=f"Research loop failed: {type(e).__name__}")


@app.get("/rag/documents")
async def list_rag_documents():
    """Returns list of indexed documents and their chunk counts."""
    from app.rag.vectorstore import vector_store
    docs = await asyncio.to_thread(vector_store.list_documents)
    total_chunks = await asyncio.to_thread(vector_store.count)
    return {"total_chunks": total_chunks, "documents": docs}


@app.post("/ingest")
async def trigger_ingest(background_tasks: BackgroundTasks):
    if _ingestion_status["running"]:
        return {"status": "ingestion already in progress"}

    background_tasks.add_task(_run_ingestion)
    return {"status": "ingestion started in background"}



async def _run_ingestion():
    _ingestion_status["running"] = True
    try:
        await asyncio.to_thread(ingest_module.ingest_all)
        _ingestion_status["last_result"] = "success"
        response_cache.invalidate_all()
        logger.info("Document ingestion completed successfully")
    except Exception as e:
        _ingestion_status["last_result"] = f"error: {e}"
        logger.exception("Document ingestion failed")
    finally:
        _ingestion_status["running"] = False


@app.get("/cache/stats")
async def cache_stats():
    return response_cache.stats


@app.post("/cache/invalidate")
async def cache_invalidate():
    count = response_cache.invalidate_all()
    return {"invalidated": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
