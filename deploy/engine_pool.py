import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Engine:
    engine_id: str
    engine_type: str  # "model" or "workflow"
    project_id: str
    engine: Any  # nndeploy model or pipeline instance
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    inference_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self):
        self.last_used_at = time.time()
        self.inference_count += 1


class EnginePool:
    def __init__(self, max_engines: int = 10):
        self.max_engines = max_engines
        self._engines: OrderedDict[str, Engine] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, engine_id: str) -> Optional[Engine]:
        async with self._lock:
            engine = self._engines.get(engine_id)
            if engine:
                engine.touch()
                # Move to end (most recently used)
                self._engines.move_to_end(engine_id)
            return engine

    async def add(self, engine: Engine) -> Engine:
        async with self._lock:
            # Evict LRU if at capacity
            while len(self._engines) >= self.max_engines:
                await self._evict_lru_unlocked()

            self._engines[engine.engine_id] = engine
            return engine

    async def remove(self, engine_id: str) -> bool:
        async with self._lock:
            engine = self._engines.pop(engine_id, None)
            if engine:
                await self._release_engine(engine)
                return True
            return False

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._engines)
            for engine in list(self._engines.values()):
                await self._release_engine(engine)
            self._engines.clear()
            return count

    async def list_engines(self) -> list:
        async with self._lock:
            return [
                {
                    "engine_id": e.engine_id,
                    "type": e.engine_type,
                    "project_id": e.project_id,
                    "loaded_at": e.loaded_at,
                    "last_used_at": e.last_used_at,
                    "inference_count": e.inference_count,
                }
                for e in self._engines.values()
            ]

    async def _evict_lru_unlocked(self):
        if not self._engines:
            return
        # Pop first item (least recently used)
        engine_id, engine = self._engines.popitem(last=False)
        await self._release_engine(engine)

    async def _release_engine(self, engine: Engine):
        # Release nndeploy resources
        try:
            if hasattr(engine.engine, "release"):
                engine.engine.release()
            elif hasattr(engine.engine, "__del__"):
                del engine.engine
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._engines)
