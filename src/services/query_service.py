from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
from datetime import datetime
import wandb
import json
from openai import AsyncOpenAI
from functools import lru_cache
import asyncio
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    supporting_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]

class WeaveQueryService:
    def __init__(self, project_name: str, openai_api_key: str):
        self.api = wandb.Api()
        self.project_name = project_name
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self._cached_runs = None
        self._last_cache_update = None
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Pre-compile metric paths for faster lookup
        self.metric_paths = {
            'accuracy': 'eval/accuracy',
            'precision': 'eval/precision',
            'recall': 'eval/recall',
            'f1': 'eval/f1',
            'loss': 'eval/loss',
            'train_loss': 'train/loss',
            'train_epoch': 'train/epoch',
            'learning_rate': 'train/learning_rate'
        }
        
        # Cache common query patterns
        self.query_patterns = {
            "compare": ["compare", "versus", "vs", "difference"],
            "ranking": ["best", "top", "highest", "lowest", "worst"],
            "explanation": ["why", "how", "explain", "what is"],
            "recommendation": ["suggest", "recommend", "should"]
        }

    @lru_cache(maxsize=100)
    def _determine_query_type(self, query: str) -> str:
        """Cached query type determination"""
        query_lower = query.lower()
        for qtype, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return qtype
        return "general"

    async def _get_cached_runs(self) -> Dict[str, Any]:
        """Get cached runs data or refresh if expired"""
        current_time = datetime.now().timestamp()
        
        if (self._cached_runs is None or 
            self._last_cache_update is None or 
            current_time - self._last_cache_update > self.cache_ttl):
            
            runs = self.api.runs(self.project_name)
            self._cached_runs = await self._process_runs(runs)
            self._last_cache_update = current_time
            
        return self._cached_runs

    async def _process_runs(self, runs) -> Dict[str, Any]:
        """Asynchronously process W&B runs"""
        processed_data = {
            "experiments": [],
            "available_metrics": set(),
            "hyperparameters": set()
        }
        
        # Process runs in parallel
        async def process_single_run(run):
            if not hasattr(run, 'summary') or run.state != "finished":
                return None
                
            summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else run.summary
            
            metrics = {}
            for metric_name, path in self.metric_paths.items():
                value = summary.get(path)
                if value is not None:
                    metrics[metric_name] = (
                        value['true_fraction'] if isinstance(value, dict) and 'true_fraction' in value
                        else value
                    )
            
            if not metrics:
                return None
                
            config = {
                key: run.config[key]
                for key in ['learning_rate', 'warmup_ratio', 'num_train_epochs']
                if key in run.config
            }
            
            return {
                "id": run.id,
                "name": run.name,
                "metrics": metrics,
                "config": config,
                "status": run.state
            }

        # Process runs concurrently
        tasks = [process_single_run(run) for run in runs]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        metrics_set = set()
        hyperparams_set = set()
        
        for result in results:
            if result:
                processed_data["experiments"].append(result)
                metrics_set.update(result["metrics"].keys())
                hyperparams_set.update(result["config"].keys())
        
        processed_data["available_metrics"] = list(metrics_set)
        processed_data["hyperparameters"] = list(hyperparams_set)
        
        return processed_data

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query with optimized data handling"""
        try:
            # Get cached runs data
            runs_data = await self._get_cached_runs()
            
            # Prepare context with pre-processed data
            query_context = {
                "available_data": runs_data,
                "metric_paths": self.metric_paths,
                "user_context": request.context or {}
            }

            # Use a more focused system prompt based on query type
            query_type = self._determine_query_type(request.query)
            focused_prompt = self._get_focused_prompt(query_type)

            # Generate response using LLM
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview", # gpt-4-turbo-preview
                messages=[
                    {"role": "system", "content": focused_prompt},
                    {"role": "user", "content": f"Based on this data:\n{json.dumps(query_context, indent=2)}\n\nQuestion: {request.query}"}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            return QueryResponse(
                answer=response.choices[0].message.content,
                supporting_data=runs_data,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "query_type": query_type,
                    "project_name": self.project_name,
                    "experiment_count": len(runs_data["experiments"]),
                    "metrics_analyzed": runs_data["available_metrics"]
                }
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._generate_error_response(request.query, str(e))

    @lru_cache(maxsize=5)
    def _get_focused_prompt(self, query_type: str) -> str:
        """Get cached focused system prompt based on query type"""
        base_prompt = """You are an AI assistant specialized in analyzing ML experiment results, particularly for hallucination detection models."""
        
        prompts = {
            "comparison": base_prompt + " Focus on comparing metrics and tradeoffs between experiments.",
            "ranking": base_prompt + " Focus on identifying top/bottom performers and their characteristics.",
            "explanation": base_prompt + " Focus on explaining relationships between metrics and parameters.",
            "recommendation": base_prompt + " Focus on suggesting improvements based on observed patterns.",
            "general": base_prompt + " Provide a balanced analysis of the relevant metrics and findings."
        }
        
        return prompts.get(query_type, base_prompt)

    def _generate_error_response(self, query: str, error: str) -> QueryResponse:
        """Generate standardized error response"""
        return QueryResponse(
            answer=f"I encountered an issue while processing your query about '{query}'. "
                  f"This project contains SmolLM2 models trained for hallucination detection.",
            supporting_data={"error": error},
            metadata={
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "project_name": self.project_name
            }
        )
