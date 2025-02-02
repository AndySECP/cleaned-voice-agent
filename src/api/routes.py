import logging
import wave
import io
from datetime import datetime
from base64 import b64encode

from fastapi import APIRouter, HTTPException, Depends, Request

from src.services.query_service import (
    WeaveQueryService,
    QueryRequest,
    QueryResponse
)

from src.core.agent.llm_agent import HallucinationAnalysisAgent
from src.services.data.data_manager import HallucinationDataManager
from src.core.agent.tools import AnalysisTools
from src.core.agent.memory import MemoryManager
from src.services.query.optimizer import QueryOptimizer
from src.core.config import Settings

from pydantic import BaseModel
from functools import lru_cache

import logging

logger = logging.getLogger(__name__)
router = APIRouter()


##
#  Dependencies
##

@lru_cache()
def get_settings():
    return Settings()

async def get_data_manager(request: Request):
    """Get the cached data manager from app state."""
    try:
        return request.app.state.data_manager
    except Exception as e:
        logger.error(f"Error getting data manager: {e}")
        raise HTTPException(status_code=500, detail="Data manager not initialized")

def get_memory_manager(request: Request):  # Remove async
    """Get the cached memory manager from app state"""
    try:
        return request.app.state.memory_manager
    except Exception as e:
        logger.error(f"Error getting memory manager: {e}")
        raise HTTPException(status_code=500, detail="Memory manager not initialized")

async def get_query_optimizer(data_manager=Depends(get_data_manager)):
    return QueryOptimizer(data_manager)

async def get_tools(data_manager=Depends(get_data_manager)):
    return AnalysisTools(data_manager)

async def get_agent(
    request: Request,  # Add request parameter
    data_manager=Depends(get_data_manager),
    query_optimizer=Depends(get_query_optimizer),
    tools=Depends(get_tools),
) -> HallucinationAnalysisAgent:
    settings = get_settings()
    memory_manager = get_memory_manager(request)  # Get memory manager using request
    return HallucinationAnalysisAgent(
        openai_api_key=settings.OPENAI_API_KEY,
        data_manager=data_manager,
        query_optimizer=query_optimizer,
        tools=tools,
        memory_manager=memory_manager
    )

async def get_query_service():
    settings = get_settings()
    return WeaveQueryService(
        project_name=settings.PROJECT_NAME,
        openai_api_key=settings.OPENAI_API_KEY
    )

##
#  Routes
##

@router.get("/search_memory")
async def search_memory(
    query: str,
    limit: int = 5,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Search through conversation history"""
    try:
        results = memory_manager.search_memory(query, limit)
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error searching conversation history"
        )

@router.post("/query", response_model=QueryResponse)
async def query_weave(
    request: QueryRequest,
    service: WeaveQueryService = Depends(get_query_service)
) -> QueryResponse:
    """Process a query about the Weave project"""
    try:
        return await service.process_query(request)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            answer="I encountered an issue while processing your query...",
            supporting_data={"error": str(e)},
            metadata={"status": "error", "error": str(e)}
        )

@router.post("/query_agent")
async def process_query(
    request: QueryRequest,
    current_request: Request,
    agent: HallucinationAnalysisAgent = Depends(get_agent)
):
    try:
        result = await agent.process_query(request.query, request.context)
        return QueryResponse(
            answer=result["response"],
            metadata=result["metadata"],
            supporting_data={}
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            answer="I encountered an error processing your query. Please try again.",
            metadata={"error": str(e)},
            supporting_data={}
        )

@router.get("/session")
async def get_session():
    response = await httpx.post(
        "https://api.openai.com/v1/realtime/sessions",
        headers={
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-realtime-preview-2024-12-17",
            "voice": "alloy"  # or your preferred voice
        }
    )
    return response.json()
