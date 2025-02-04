import logging
import wave
import io
from datetime import datetime
from base64 import b64encode
import httpx

from fastapi import APIRouter, HTTPException, Depends, Request

from src.services.query_service import (
    QueryRequest,
    QueryResponse
)

from src.core.agent.llm_agent import HallucinationAnalysisAgent
from src.services.data.data_manager import HallucinationDataManager
from src.core.agent.tools import AnalysisTools
from src.core.agent.memory import MemoryManager
from src.services.query.optimizer import QueryOptimizer
from src.core.config import Settings
from src.api.models import FunctionExecuteRequest

from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from functools import lru_cache
import numpy as np

import logging
import json
import math

logger = logging.getLogger(__name__)
router = APIRouter()

def clean_nan_values(data):
    """Recursively clean NaN values in nested structures"""
    if isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nan_values(x) for x in data]
    elif isinstance(data, (float, np.float64, np.float32)):
        if math.isnan(data):
            return "NaN"
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        return float(data)
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return clean_nan_values(data.tolist())
    return data

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
        memory_manager=memory_manager,
        use_realtime=True
    )

##
#  Routes
##

@router.get("/search_memory") # not used
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
            supporting_data={"session": result.get("session")}  # Include session for realtime
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
    """Generate an ephemeral token for WebRTC connection"""
    settings = get_settings()
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": "alloy"
                },
                timeout=30.0
            )
            
            response.raise_for_status()
            return response.json()  # FastAPI will validate this against SessionResponse
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error during session token generation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating session token: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during session token generation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )

@router.post("/execute_function")
async def execute_function(
    request: FunctionExecuteRequest,
    agent: HallucinationAnalysisAgent = Depends(get_agent)
):
    """Execute a specific function using the agent's tools"""
    try:
        if request.function_name not in agent.available_functions:
            raise HTTPException(
                status_code=404,
                detail=f"Function {request.function_name} not found"
            )

        # Execute the function using the agent's tools
        result = await agent.available_functions[request.function_name](**request.arguments)
        
        # Clean the result of NaN values
        cleaned_result = clean_nan_values(result)
        
        # Return the cleaned result
        return {
            "status": "success",
            "data": cleaned_result,
            "call_id": request.call_id
        }
    except Exception as e:
        logger.error(f"Error executing function {request.function_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
