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
    """Get the cached data manager from app state (for HTTP routes)."""
    try:
        return request.app.state.data_manager
    except Exception as e:
        logger.error(f"Error getting data manager: {e}")
        raise HTTPException(status_code=500, detail="Data manager not initialized")

async def get_query_optimizer(data_manager=Depends(get_data_manager)):
    return QueryOptimizer(data_manager)

async def get_tools(data_manager=Depends(get_data_manager)):
    return AnalysisTools(data_manager)

async def get_agent(
    data_manager=Depends(get_data_manager),
    query_optimizer=Depends(get_query_optimizer),
    tools=Depends(get_tools)
) -> HallucinationAnalysisAgent:
    settings = get_settings()
    return HallucinationAnalysisAgent(
        openai_api_key=settings.OPENAI_API_KEY,
        data_manager=data_manager,
        query_optimizer=query_optimizer,
        tools=tools
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
    agent: HallucinationAnalysisAgent = Depends(get_agent)
):
    """Process a query using the LLM agent with cached data"""
    try:
        result = await agent.process_query(request.query, request.context)
        return QueryResponse(
            answer=result.get("response", "I couldn't process your query at this time."),
            metadata=result.get("metadata", {}),
            supporting_data=result.get("raw_results", {})
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            answer="I encountered an error processing your query. Please try again.",
            metadata={"error": str(e)},
            supporting_data={}
        )
