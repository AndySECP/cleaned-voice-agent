# services/data/cache_manager.py
from typing import Dict, Any, Optional
import redis
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=True
            )
            self._memory_cache: Dict[str, Dict] = {}
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache (first memory, then Redis)"""
        # Try memory cache first
        if key in self._memory_cache:
            cached = self._memory_cache[key]
            if datetime.now() < cached['expiry']:
                return cached['data']
            else:
                del self._memory_cache[key]

        # Try Redis if memory cache miss
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    # Update memory cache
                    parsed_data = json.loads(data)
                    self._memory_cache[key] = {
                        'data': parsed_data,
                        'expiry': datetime.now() + timedelta(minutes=5)
                    }
                    return parsed_data
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set data in both memory and Redis cache"""
        try:
            # Set memory cache
            self._memory_cache[key] = {
                'data': value,
                'expiry': datetime.now() + timedelta(seconds=ttl_seconds)
            }

            # Set Redis cache if available
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    ttl_seconds,
                    json.dumps(value)
                )
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def delete(self, key: str):
        """Delete data from both caches"""
        if key in self._memory_cache:
            del self._memory_cache[key]

        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
                