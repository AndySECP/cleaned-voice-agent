# core/agent/tools.py
from typing import Dict, List, Any
import pandas as pd
import logging
from src.services.data.data_manager import METRIC_MAPPING

logger = logging.getLogger(__name__)

class AnalysisTools:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def _get_metric_column(self, friendly_name: str) -> str:
        """Convert friendly metric name to actual column name"""
        # If it's already a column name, return it
        if friendly_name in self.data_manager.df.columns:
            return friendly_name
        # Otherwise, try to map it
        return METRIC_MAPPING.get(friendly_name, friendly_name)

    async def get_best_models(
        self, 
        metric: str = 'accuracy',
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Get top performing models for a specific metric"""
        try:
            metric_col = self._get_metric_column(metric)
            best_models = self.data_manager.get_best_models(metric_col, top_k)
            
            if not best_models:
                return {
                    'status': 'error',
                    'error': f'No data available for metric: {metric}'
                }
            
            return {
                'status': 'success',
                'data': best_models,
                'metric': metric
            }
        except Exception as e:
            logger.error(f"Error in get_best_models: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def compare_hyperparams(
        self, 
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Compare performance across different hyperparameter settings"""
        try:
            metric_col = self._get_metric_column(metric)
            comparisons = self.data_manager.compare_hyperparams(metric_col)
            return {
                'status': 'success',
                'data': comparisons,
                'metric': metric
            }
        except Exception as e:
            logger.error(f"Error in compare_hyperparams: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def get_experiment_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific experiment"""
        try:
            details = self.data_manager.get_experiment_details(run_id)
            if details is None:
                return {
                    'status': 'error',
                    'error': f'Run {run_id} not found'
                }
                
            # Convert metric columns to friendly names
            friendly_metrics = {}
            reverse_mapping = {v: k for k, v in METRIC_MAPPING.items()}
            
            for key, value in details.items():
                if key in reverse_mapping:
                    friendly_metrics[reverse_mapping[key]] = value
                else:
                    friendly_metrics[key] = value
                    
            return {
                'status': 'success',
                'data': friendly_metrics
            }
        except Exception as e:
            logger.error(f"Error in get_experiment_details: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Available tools for the LLM agent
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_best_models",
            "description": "Get top performing models for a specific metric",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to rank by (e.g., accuracy, f1, precision, recall)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top models to return"
                    }
                },
                "required": ["metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_hyperparams",
            "description": "Compare performance across different hyperparameter settings",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to use for comparison (e.g., accuracy, f1)"
                    }
                },
                "required": ["metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_details",
            "description": "Get detailed information about a specific experiment",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "ID of the experiment run"
                    }
                },
                "required": ["run_id"]
            }
        }
    }
]
