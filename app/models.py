from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class TaskKind(Enum):
    TRANSLATE = "draft"
    EXTRACT = "extract"
    ALIGN = "align"
    TRAIN = "train"


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class Task:
    id: str
    kind: TaskKind
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.QUEUED
    script_path: Optional[str] = None
