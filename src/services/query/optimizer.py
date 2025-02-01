# core/agent/tools.py
from typing import Dict, List, Any
import pandas as pd
from src.services.data.data_manager import HallucinationDataManager

class QueryOptimizer:
    def __init__(self, data_manager: HallucinationDataManager):
        self.data_manager = data_manager
        self.query_patterns = {
            'best_model': ['best', 'top', 'highest'],
            'comparison': ['compare', 'versus', 'vs'],
            'trend': ['trend', 'over time', 'progression'],
            'correlation': ['correlation', 'relationship', 'impact']
        }
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Optimize query execution path based on query pattern"""
        query_type = self._identify_query_type(query)
        
        optimizers = {
            'best_model': self._optimize_best_model_query,
            'comparison': self._optimize_comparison_query,
            'trend': self._optimize_trend_query,
            'correlation': self._optimize_correlation_query
        }
        
        return optimizers.get(query_type, self._optimize_general_query)(query)
    
    def _identify_query_type(self, query: str) -> str:
        """Identify query type for optimization"""
        query_lower = query.lower()
        for qtype, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return qtype
        return 'general'
    
    def _optimize_best_model_query(self, query: str) -> Dict[str, Any]:
        """Optimize queries about best performing models"""
        metric = 'eval.accuracy.true_fraction'  # default
        if 'hallucination' in query.lower():
            metric = 'eval.hallucination_rate'
        elif 'f1' in query.lower():
            metric = 'eval.f1'
        
        return {
            'query_type': 'best_model',
            'metric': metric,
            'cache_key': f'best_models_{metric}',
            'required_columns': ['run_id', 'config.model', metric]
        }
