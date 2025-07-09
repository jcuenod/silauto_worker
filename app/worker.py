import logging
import time
import requests
from typing import Optional

from app.gpu_checker import GPUChecker
from app.models import Task, TaskKind, TaskStatus
from app.task_executor import TaskExecutor

from .env import (
    SILAUTO_URL,
)


class SilAutoWorker:
    """Main worker class that fetches and executes tasks"""

    def __init__(self):
        self.base_url = SILAUTO_URL.rstrip("/")
        self.logger = logging.getLogger(f"{__name__}.SilAutoWorker")
        self.session = requests.Session()

        self.logger.info("Worker initialized and ready to process tasks")

    def fetch_next_task(self) -> Optional[Task]:
        """Fetch the next available task from the API"""
        try:
            # Check GPU availability before requesting tasks
            gpu_available = GPUChecker.is_gpu_id_idle()
            if not gpu_available:
                self.logger.debug("GPU unavailable")
                return None

            # Include GPU status in the request to help server decide which tasks to assign
            url = f"{self.base_url}/tasks/next"
            self.logger.debug(
                f"Fetching next task from: {url} (GPU available: {gpu_available})"
            )

            response = self.session.get(url, timeout=30)

            if response.status_code == 404:
                self.logger.debug("No tasks available (404)")
                return None

            response.raise_for_status()
            data = response.json()

            # Parse task data - server sends kind instead of type
            task_kind = TaskKind(data["kind"])
            task = Task(
                id=data["id"], kind=task_kind, parameters=data.get("parameters", {})
            )

            self.logger.info(
                f"Fetched task: {task.id} (kind: {task.kind.value}) - GPU available: {gpu_available}"
            )
            return task

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching next task: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing task data: {e}")
            return None

    def update_task_status(
        self, task_id: str, status: TaskStatus, message: str = ""
    ) -> bool:
        """Update the status of a task"""
        # Prepare payload
        payload = {
            "status": status.value,
            "error": message if status == TaskStatus.FAILED else None,
        }

        # Add timestamp fields based on status
        if status == TaskStatus.RUNNING:
            payload["started_at"] = time.time()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            payload["ended_at"] = time.time()

        try:
            # Since there's no explicit update endpoint in the OpenAPI spec,
            # let's try POST to a status endpoint first
            url = f"{self.base_url}/tasks/{task_id}/status"

            self.logger.debug(f"Updating task {task_id} status to {status.value}")

            response = self.session.patch(url, json=payload, timeout=30)
            response.raise_for_status()

            self.logger.info(
                f"Successfully updated task {task_id} status to {status.value}"
            )
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error updating task {task_id} status: {e}")
            return False

    def process_task(self, task: Task) -> bool:
        """Process a single task"""
        # Update status to RUNNING
        if not self.update_task_status(task.id, TaskStatus.RUNNING):
            self.logger.error(f"Failed to update task {task.id} to RUNNING status")
            return False

        # Execute the task
        executor = TaskExecutor(task)
        success = executor.execute()

        # Update final status
        final_status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        message = "Task completed successfully" if success else "Task execution failed"

        self.update_task_status(task.id, final_status, message)

        return success

    def run(self):
        """Main worker loop"""
        self.logger.info("Starting SILAUTO worker...")

        while True:
            try:
                # Fetch next task
                task = self.fetch_next_task()

                if task is None:
                    # No tasks available, wait and try again
                    self.logger.debug("No tasks available, waiting 60 seconds...")
                    time.sleep(60)
                    continue

                # Process the task
                self.process_task(task)

                # Brief pause between tasks
                time.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("Worker stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in worker loop: {e}")
                time.sleep(10)  # Wait before retrying
