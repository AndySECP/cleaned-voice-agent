from pydantic import BaseModel
from typing import Optional, Dict, Any

class AudioQuery(BaseModel):
    audio_content: bytes

class TextQuery(BaseModel):
    text: str

class QueryAudioResponse(BaseModel):
    text: str
    audio_url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class MetricsSummary(BaseModel):
    metric_name: str
    value: float
    comparison: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    metrics: List[MetricsSummary]
    metadata: Dict[str, Any]