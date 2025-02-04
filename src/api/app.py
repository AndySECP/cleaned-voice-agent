from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.services.data.data_manager import HallucinationDataManager
from src.core.agent.memory import MemoryManager
from src.services.query_service import WeaveQueryService
from functools import lru_cache
from src.core.config import Settings
import logging

from fastapi.responses import JSONResponse
import httpx

logger = logging.getLogger(__name__)

app = FastAPI(title="Weave Query API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router with prefix
app.include_router(router, prefix="/api/v1")


@lru_cache()
def get_settings():
    return Settings()


@app.get("/")
async def root():
    return {"message": "Welcome to Weave Query API"}


@app.post("/get_token")
async def get_ephemeral_token():
    """Generate an ephemeral token from OpenAI"""
    settings = get_settings()
    try:
        logger.info("Requesting ephemeral token from OpenAI")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": "gpt-4o-realtime-preview-2024-12-17", "voice": "alloy"},
            )

            response_data = response.json()
            logger.info(f"OpenAI Response Status: {response.status_code}")

            if response.status_code != 200:
                return JSONResponse(
                    content={"error": "Failed to get token", "details": response_data},
                    status_code=response.status_code,
                )

            # Extract token from client_secret
            if (
                "client_secret" in response_data
                and "value" in response_data["client_secret"]
            ):
                token = response_data["client_secret"]["value"]
                logger.info("Successfully extracted token from response")
                return JSONResponse(content={"token": token})
            else:
                logger.error("Unexpected response structure from OpenAI")
                return JSONResponse(
                    content={
                        "error": "Invalid response structure",
                        "details": response_data,
                    },
                    status_code=500,
                )

    except Exception as e:
        logger.exception("Error getting token")
        return JSONResponse(
            content={"error": "Server error", "details": str(e)}, status_code=500
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    if isinstance(exc, HTTPException):
        raise exc
    return {"detail": str(exc)}


@app.on_event("startup")
async def startup_event():
    """Initialize data and memory at startup"""
    try:
        # Initialize data manager
        data_manager = HallucinationDataManager()
        await data_manager.initialize()

        # Initialize memory manager (no async needed here)
        memory_manager = MemoryManager(
            cache_size=20,  # Keep 20 most recent messages in memory
            persist_after=5,  # Write to database every 5 messages
        )

        # Store in app state for access across requests
        app.state.data_manager = data_manager
        app.state.memory_manager = memory_manager

        logger.info("Successfully initialized data manager and memory at startup")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
