from pydantic import BaseModel
from typing import Optional, Dict, Any

class FunctionExecuteRequest(BaseModel):
    function_name: str
    arguments: Dict[str, Any]
    call_id: str
