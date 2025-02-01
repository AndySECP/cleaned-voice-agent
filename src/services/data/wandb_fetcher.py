from typing import Dict, List, Any, Optional
import pandas as pd
import redis
import json
from functools import lru_cache
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import wandb
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class MetricCache:
    data: Dict[str, Any]
    timestamp: datetime
    ttl: int = 300  # 5 minutes default TTL

    def is_valid(self) -> bool:
        return datetime.now() - self.timestamp < timedelta(seconds=self.ttl)

class WandbDataFetcher:
    """Handle W&B data fetching and processing"""
    def __init__(self, entity: str = "c-metrics", project: str = "hallucination"):
        self.entity = entity
        self.project = project
        self.api = wandb.Api()
        
    def _parse_nested_value(self, value: Any) -> Dict[str, Any]:
        """Parse potentially nested values in W&B data"""
        if not isinstance(value, str):
            return {'value': value}
            
        try:
            parsed = json.loads(value.replace("'", '"'))
            return parsed if isinstance(parsed, dict) else {'value': parsed}
        except:
            return {'value': value}

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested metrics dictionary"""
        flattened = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                if 'value' in value:
                    flattened[f"{prefix}{key}"] = value['value']
                else:
                    flattened.update(self._flatten_metrics(value, f"{prefix}{key}."))
            else:
                flattened[f"{prefix}{key}"] = value
        return flattened

    async def fetch_data(self) -> pd.DataFrame:
        """Fetch and process W&B data asynchronously"""
        try:
            logger.info(f"Fetching data from {self.entity}/{self.project}")
            
            # Run W&B API calls in a thread pool to avoid blocking
            with ThreadPoolExecutor() as executor:
                runs = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: list(self.api.runs(f"{self.entity}/{self.project}"))
                )
            
            logger.info(f"Found {len(runs)} runs")
            all_runs_data = []
            
            for run in runs:
                # Basic run info
                run_data = {
                    'run_id': run.id,
                    'run_name': run.name,
                    'run_state': run.state,
                    'created_at': run.created_at
                }
                
                # Process config
                for key, value in run.config.items():
                    parsed = self._parse_nested_value(value)
                    if isinstance(parsed, dict) and 'value' not in parsed:
                        flattened = self._flatten_metrics(parsed, f'config.{key}.')
                        run_data.update(flattened)
                    else:
                        run_data[f'config.{key}'] = parsed.get('value', value)
                
                # Process metrics/summary
                if run.summary:
                    for key, value in run.summary._json_dict.items():
                        if key.startswith('_'):
                            continue
                            
                        if key.startswith('eval/') or key.startswith('train/'):
                            clean_key = key.replace('/', '.')
                            if key == 'eval/accuracy':
                                try:
                                    parsed = (value if isinstance(value, dict) 
                                            else self._parse_nested_value(value))
                                    if isinstance(parsed, dict):
                                        run_data['eval.accuracy.true_count'] = parsed.get('true_count')
                                        run_data['eval.accuracy.true_fraction'] = parsed.get('true_fraction')
                                    continue
                                except:
                                    run_data[clean_key] = value
                            else:
                                run_data[clean_key] = value
                
                all_runs_data.append(run_data)
            
            df = pd.DataFrame(all_runs_data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at', ascending=False)
            
            logger.info(f"Successfully processed {len(df)} runs")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching W&B data: {str(e)}")
            raise