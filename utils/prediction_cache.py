# utils/prediction_cache.py

from datetime import datetime, timedelta
import threading
from typing import Dict, Optional

class PredictionCache:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def get_prediction(self, game_id: str) -> Optional[dict]:
        with self._lock:
            if game_id in self._cache:
                prediction = self._cache[game_id]
                # Check if prediction is still valid (less than 1 hour old)
                if datetime.now() - prediction['timestamp'] < timedelta(hours=1):
                    return prediction['data']
                else:
                    del self._cache[game_id]
            return None

    def store_prediction(self, game_id: str, prediction_data: dict):
        with self._lock:
            self._cache[game_id] = {
                'data': prediction_data,
                'timestamp': datetime.now()
            }

    def clear_old_predictions(self):
        with self._lock:
            current_time = datetime.now()
            expired_keys = [
                k for k, v in self._cache.items()
                if current_time - v['timestamp'] > timedelta(hours=1)
            ]
            for k in expired_keys:
                del self._cache[k]



