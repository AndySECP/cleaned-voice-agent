# core/agent/tools.py
from typing import Dict, List, Any, Optional
import pandas as pd
import logging
from src.services.data.data_manager import METRIC_MAPPING

logger = logging.getLogger(__name__)

class AnalysisTools:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.DEFAULT_METRICS = ['eval.f1', 'eval.precision', 'eval.recall', 'eval.hallucination_rate']
        self.CONFIG_COLUMNS = [col for col in self.data_manager.df.columns if col.startswith('config.')]

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
    
    async def analyze_by_model_type(
        self,
        metrics: Optional[List[str]] = None,
        group_by: str = 'config.model_type'
    ) -> Dict[str, Any]:
        """
        Analyze performance metrics grouped by model type or other grouping variable.
        
        Args:
            metrics: List of metrics to analyze. If None, uses default metrics
            group_by: Column to group by (default: config.model_type)
            
        Returns:
            Dictionary with analysis results including means, counts, and distributions
        """
        try:
            df = self.data_manager.df
            metrics = metrics or self.DEFAULT_METRICS
            
            # Validate metrics exist
            invalid_metrics = [m for m in metrics if m not in df.columns]
            if invalid_metrics:
                return {
                    'status': 'error',
                    'error': f'Invalid metrics: {invalid_metrics}'
                }
                
            # Group analysis
            grouped_stats = df.groupby(group_by)[metrics].agg([
                'mean', 'std', 'count', 
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75)
            ]).round(4)
            
            # Calculate proportions of each group
            total_runs = len(df)
            proportions = (df[group_by].value_counts() / total_runs * 100).round(2)
            
            return {
                'status': 'success',
                'data': {
                    'statistics': grouped_stats.to_dict(),
                    'proportions': proportions.to_dict(),
                    'total_runs': total_runs
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_by_model_type: {e}")
            return {'status': 'error', 'error': str(e)}

    async def analyze_config_impact(
        self,
        target_metric: str = 'eval.f1',
        config_params: Optional[List[str]] = None,
        method: str = 'correlation'
    ) -> Dict[str, Any]:
        """
        Analyze how different configurations impact model performance.
        
        Args:
            target_metric: Metric to analyze impact on
            config_params: List of config parameters to analyze. If None, uses all numeric configs
            method: Analysis method ('correlation' or 'feature_importance')
            
        Returns:
            Dictionary with impact analysis results
        """
        try:
            df = self.data_manager.df
            
            # Validate target metric
            if target_metric not in df.columns:
                return {
                    'status': 'error',
                    'error': f'Invalid target metric: {target_metric}'
                }
            
            # Get numeric config columns if not specified
            if config_params is None:
                config_params = [col for col in self.CONFIG_COLUMNS 
                               if pd.api.types.is_numeric_dtype(df[col])]
            
            # Calculate correlations
            correlations = {}
            for param in config_params:
                if param in df.columns and pd.api.types.is_numeric_dtype(df[param]):
                    corr = df[param].corr(df[target_metric])
                    if not pd.isna(corr):
                        correlations[param] = round(corr, 4)
            
            # Sort by absolute correlation value
            correlations = dict(sorted(correlations.items(), 
                                     key=lambda x: abs(x[1]), 
                                     reverse=True))
            
            # Calculate basic stats for each parameter
            param_stats = {}
            for param in config_params:
                if param in df.columns:
                    param_stats[param] = {
                        'mean': df[param].mean() if pd.api.types.is_numeric_dtype(df[param]) else None,
                        'unique_values': df[param].nunique(),
                        'most_common': df[param].mode().iloc[0] if len(df[param].mode()) > 0 else None
                    }
            
            return {
                'status': 'success',
                'data': {
                    'correlations': correlations,
                    'parameter_stats': param_stats,
                    'target_metric': target_metric
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_config_impact: {e}")
            return {'status': 'error', 'error': str(e)}

    async def get_performance_distribution(
        self,
        metrics: Optional[List[str]] = None,
        percentiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ) -> Dict[str, Any]:
        """
        Get statistical distribution of performance metrics.
        
        Args:
            metrics: List of metrics to analyze. If None, uses default metrics
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary with distribution statistics
        """
        try:
            df = self.data_manager.df
            metrics = metrics or self.DEFAULT_METRICS
            
            # Validate metrics
            invalid_metrics = [m for m in metrics if m not in df.columns]
            if invalid_metrics:
                return {
                    'status': 'error',
                    'error': f'Invalid metrics: {invalid_metrics}'
                }
            
            distributions = {}
            for metric in metrics:
                values = df[metric].dropna()
                
                distributions[metric] = {
                    'mean': round(values.mean(), 4),
                    'std': round(values.std(), 4),
                    'min': round(values.min(), 4),
                    'max': round(values.max(), 4),
                    'percentiles': {
                        str(int(p * 100)): round(values.quantile(p), 4)
                        for p in percentiles
                    }
                }
            
            return {
                'status': 'success',
                'data': distributions
            }
        except Exception as e:
            logger.error(f"Error in get_performance_distribution: {e}")
            return {'status': 'error', 'error': str(e)}

    async def compare_architectures(
        self,
        metrics: Optional[List[str]] = None,
        arch_column: str = 'config.architectures',
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Compare different model architectures with statistical tests.
        
        Args:
            metrics: List of metrics to compare. If None, uses default metrics
            arch_column: Column containing architecture information
            min_samples: Minimum samples required for comparison
            
        Returns:
            Dictionary with comparison results and statistical tests
        """
        try:
            df = self.data_manager.df
            metrics = metrics or self.DEFAULT_METRICS
            
            # Validate metrics
            invalid_metrics = [m for m in metrics if m not in df.columns]
            if invalid_metrics:
                return {
                    'status': 'error',
                    'error': f'Invalid metrics: {invalid_metrics}'
                }
            
            # Get architectures with sufficient samples
            arch_counts = df[arch_column].value_counts()
            valid_archs = arch_counts[arch_counts >= min_samples].index
            
            if len(valid_archs) < 2:
                return {
                    'status': 'error',
                    'error': f'Need at least 2 architectures with {min_samples}+ samples'
                }
            
            results = {}
            for metric in metrics:
                arch_stats = {}
                statistical_tests = {}
                
                # Calculate basic stats for each architecture
                for arch in valid_archs:
                    values = df[df[arch_column] == arch][metric].dropna()
                    arch_stats[arch] = {
                        'mean': round(values.mean(), 4),
                        'std': round(values.std(), 4),
                        'count': len(values)
                    }
                
                # Perform statistical tests between architectures
                for i, arch1 in enumerate(valid_archs):
                    for arch2 in valid_archs[i+1:]:
                        values1 = df[df[arch_column] == arch1][metric].dropna()
                        values2 = df[df[arch_column] == arch2][metric].dropna()
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(values1, values2)
                        
                        statistical_tests[f'{arch1}_vs_{arch2}'] = {
                            't_statistic': round(t_stat, 4),
                            'p_value': round(p_value, 4),
                            'significant': p_value < 0.05
                        }
                
                results[metric] = {
                    'architecture_stats': arch_stats,
                    'statistical_tests': statistical_tests
                }
            
            return {
                'status': 'success',
                'data': results
            }
        except Exception as e:
            logger.error(f"Error in compare_architectures: {e}")
            return {'status': 'error', 'error': str(e)}

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
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_by_model_type",
            "description": "Get performance statistics grouped by model type, including means, distributions, and proportions of experiments",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to analyze (e.g., ['eval.f1', 'eval.precision']). If not provided, uses default metrics"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Column to group by (default: config.model_type)",
                        "default": "config.model_type"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_config_impact",
            "description": "Analyze how different configuration parameters impact model performance using correlation analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_metric": {
                        "type": "string",
                        "description": "Target metric to analyze impact on (e.g., eval.f1)",
                        "default": "eval.f1"
                    },
                    "config_params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of config parameters to analyze. If not provided, uses all numeric configs"
                    },
                    "method": {
                        "type": "string",
                        "description": "Analysis method ('correlation' or 'feature_importance')",
                        "enum": ["correlation", "feature_importance"],
                        "default": "correlation"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_performance_distribution",
            "description": "Get statistical distribution of performance metrics including means, standard deviations, and percentiles",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to analyze. If not provided, uses default metrics"
                    },
                    "percentiles": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of percentiles to calculate (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])",
                        "default": [0.1, 0.25, 0.5, 0.75, 0.9]
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_architectures",
            "description": "Compare different model architectures with statistical tests and performance metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to compare. If not provided, uses default metrics"
                    },
                    "arch_column": {
                        "type": "string",
                        "description": "Column containing architecture information",
                        "default": "config.architectures"
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "Minimum samples required for comparison",
                        "default": 5
                    }
                }
            }
        }
    },
]
