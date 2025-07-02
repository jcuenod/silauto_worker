#!/usr/bin/env python3
"""
SILAUTO Worker - Handles Draft (Translate), Extract, Align, and Train tasks
"""

import os
import time
import logging
import subprocess
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .env import SILAUTO_URL, SILNLP_ROOT, CUDA_DEVICE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


class GPUChecker:
    """Check GPU availability using nvidia-smi and torch"""

    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU is available using nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and lines[0]:
                    logger.info(f"GPU detected: {lines[0]}")
                    return True

            logger.warning("nvidia-smi command failed or no GPU detected")
            return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Error checking GPU: {e}")
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get detailed GPU information"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 6:
                            gpus.append(
                                {
                                    "name": parts[0],
                                    "memory_total": int(parts[1]),
                                    "memory_used": int(parts[2]),
                                    "memory_free": int(parts[3]),
                                    "utilization": int(parts[4]),
                                    "temperature": int(parts[5]),
                                }
                            )

                return {"gpus": gpus, "count": len(gpus)}

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")

        return {"gpus": [], "count": 0}

    @staticmethod
    def get_idle_gpu_id(
        memory_threshold_mb: int = 500, utilization_threshold: int = 10
    ) -> Optional[int]:
        """
        Return the index of an 'idle' GPU (low memory use and low utilization), or None if none found.
        :param memory_threshold_mb: Maximum memory used (in MB) to consider GPU idle.
        :param utilization_threshold: Maximum utilization (%) to consider GPU idle.
        :return: GPU index (int) if idle GPU is found, else None.
        """
        info = GPUChecker.get_gpu_info()
        for idx, gpu in enumerate(info.get("gpus", [])):
            if (
                gpu["memory_used"] <= memory_threshold_mb
                and gpu["utilization"] <= utilization_threshold
            ):
                logger.info(
                    f"GPU {gpu['name']} (id={idx}) is idle (memory_used={gpu['memory_used']}MB, utilization={gpu['utilization']}%)"
                )
                return idx
        logger.info("No idle GPU found")
        return None

    @staticmethod
    def is_gpu_id_idle(
        memory_threshold_mb: int = 500, utilization_threshold: int = 10
    ) -> bool:
        """
        Return the index of an 'idle' GPU (low memory use and low utilization), or None if none found.
        :param memory_threshold_mb: Maximum memory used (in MB) to consider GPU idle.
        :param utilization_threshold: Maximum utilization (%) to consider GPU idle.
        :return: GPU index (int) if idle GPU is found, else None.
        """
        if not CUDA_DEVICE:
            return False

        info = GPUChecker.get_gpu_info()
        try:
            cuda_device_id = int(CUDA_DEVICE)
            gpus = info.get("gpus", [])
            if cuda_device_id < len(gpus):
                gpu = gpus[cuda_device_id]
                return (
                    gpu
                    and gpu["memory_used"] <= memory_threshold_mb
                    and gpu["utilization"] <= utilization_threshold
                )
        except (ValueError, IndexError):
            pass

        return False


class TaskExecutor:
    """Execute different types of tasks"""

    def __init__(self, task: Task):
        self.task = task
        self.logger = logging.getLogger(f"{__name__}.TaskExecutor")

    def execute(self) -> bool:
        """Execute the task based on its kind"""
        try:
            self.logger.info(f"Executing {self.task.kind.value} task {self.task.id}")

            if self.task.kind == TaskKind.TRANSLATE:
                return self._execute_translate()
            elif self.task.kind == TaskKind.EXTRACT:
                return self._execute_extract()
            elif self.task.kind == TaskKind.ALIGN:
                return self._execute_align()
            elif self.task.kind == TaskKind.TRAIN:
                return self._execute_train()
            else:
                self.logger.error(f"Unknown task kind: {self.task.kind}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing task {self.task.id}: {e}")
            return False

    def _execute_translate(self) -> bool:
        """Execute translation task"""
        # The server sends task parameters, not script content
        # We need to generate the appropriate script based on the parameters
        try:
            params = self.task.parameters
            experiment_name = params.get("experiment_name")
            train_task_id = params.get("train_task_id")
            source_project_id = params.get("source_project_id")
            book_names = params.get("book_names", [])
            source_script_code = params.get("source_script_code")
            target_script_code = params.get("target_script_code")

            if not all(
                [
                    train_task_id,
                    experiment_name,
                    source_project_id,
                    book_names,
                    source_script_code,
                    target_script_code,
                ]
            ):
                self.logger.error("Missing required parameters for translate task")
                return False

            # Generate script content for translation
            script_content = self._generate_translate_script(
                str(train_task_id),
                str(source_project_id),
                list(book_names),
                str(source_script_code),
                str(target_script_code),
                str(experiment_name),
            )

            return self._run_script(script_content, "translate")

        except Exception as e:
            self.logger.error(f"Error in translate task: {e}")
            return False

    def _execute_extract(self) -> bool:
        """Execute extraction task"""
        try:
            params = self.task.parameters
            project_id = params.get("project_id")

            if not project_id:
                self.logger.error("Missing project_id for extract task")
                return False

            # Generate script content for extraction
            script_content = self._generate_extract_script(str(project_id))

            return self._run_script(script_content, "extract")

        except Exception as e:
            self.logger.error(f"Error in extract task: {e}")
            return False

    def _execute_align(self) -> bool:
        """Execute alignment task"""
        try:
            params = self.task.parameters
            experiment_name = params.get("experiment_name")
            target_scripture_file = params.get("target_scripture_file")
            source_scripture_files = params.get("source_scripture_files", [])

            if not all(
                [experiment_name, target_scripture_file, source_scripture_files]
            ):
                self.logger.error("Missing required parameters for align task")
                return False

            # Generate script content for alignment
            script_content = self._generate_align_script(
                str(experiment_name),
                str(target_scripture_file),
                list(source_scripture_files),
            )

            return self._run_script(script_content, "align")

        except Exception as e:
            self.logger.error(f"Error in align task: {e}")
            return False

    def _execute_train(self) -> bool:
        """Execute training task"""
        try:
            params = self.task.parameters
            experiment_name = params.get("experiment_name")

            if not experiment_name:
                self.logger.error("Missing required parameters for train task")
                return False

            print(experiment_name)

            # Generate script content for training
            script_content = self._generate_train_script(
                str(experiment_name),
            )

            return self._run_script(script_content, "train")

        except Exception as e:
            self.logger.error(f"Error in train task: {e}")
            return False

    def _generate_translate_script(
        self,
        train_task_id: str,
        source_project_id: str,
        book_names: List[str],
        source_script_code: str,
        target_script_code: str,
        experiment_name: str,
    ) -> str:
        """Generate script content for translation task"""
        books_str = ",".join(book_names)
        return f"""
# Extraction task for (train) project: {train_task_id}
echo "Starting translation task..."
echo "Train task ID: {train_task_id}"
echo "Experiment name: {experiment_name}"
echo "Source project ID: {source_project_id}"
echo "Books to translate: {books_str}"
echo "Source script code: {source_script_code}"
echo "Target script code: {target_script_code}"

cd {SILNLP_ROOT}

# Create a unique session name and files for this task
SESSION_NAME="extract_{self.task.id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"

echo "Running extraction in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting alignment process...'
    if poetry run python -m silnlp.nmt.translate {experiment_name} --src-project {source_project_id} --books {books_str} --src-iso {source_script_code} --trg-iso {target_script_code} --checkpoint best; then
        echo 'SUCCESS' > $STATUS_FILE
    else
        echo 'FAILED' > $STATUS_FILE
    fi
    rm -f $PID_FILE
"

# Give the screen session a moment to start and create the PID file
sleep 5

# Wait for completion by monitoring the PID file and status file
echo "Waiting for extraction to complete..."
while [ -f "$PID_FILE" ]; do
    sleep 30
done

# Check the final status
if [ -f "$STATUS_FILE" ] && [ "$(cat $STATUS_FILE)" = "SUCCESS" ]; then
    echo "Extraction task completed successfully"
    rm -f "$STATUS_FILE"
    exit 0
else
    echo "Extraction task failed"
    rm -f "$STATUS_FILE"
    exit 1
fi
"""

    def _generate_extract_script(self, project_id: str) -> str:
        """Generate script content for extraction task"""
        return f"""
# Extraction task for project: {project_id}
echo "Starting extraction task..."
echo "Project ID: {project_id}"

cd {SILNLP_ROOT}

# Create a unique session name and files for this task
SESSION_NAME="extract_{self.task.id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"

echo "Running extraction in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting alignment process...'
    if poetry run python -m silnlp.common.extract_corpora {project_id}; then
        echo 'SUCCESS' > $STATUS_FILE
    else
        echo 'FAILED' > $STATUS_FILE
    fi
    rm -f $PID_FILE
"

# Give the screen session a moment to start and create the PID file
sleep 5

# Wait for completion by monitoring the PID file and status file
echo "Waiting for extraction to complete..."
while [ -f "$PID_FILE" ]; do
    sleep 30
done

# Check the final status
if [ -f "$STATUS_FILE" ] && [ "$(cat $STATUS_FILE)" = "SUCCESS" ]; then
    echo "Extraction task completed successfully"
    rm -f "$STATUS_FILE"
    exit 0
else
    echo "Extraction task failed"
    rm -f "$STATUS_FILE"
    exit 1
fi
"""

    def _generate_align_script(
        self,
        experiment_name: str,
        target_scripture_file: str,
        source_scripture_files: List[str],
    ) -> str:
        """Generate script content for alignment task"""
        sources_str = " ".join(source_scripture_files)
        return f"""
# Alignment task
echo "Starting alignment task..."
echo "Target scripture file: {target_scripture_file}"
echo "Source scripture files: {sources_str}"

cd {SILNLP_ROOT}

# Create a unique session name and files for this task
SESSION_NAME="align_{self.task.id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"

echo "Running alignment in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting alignment process...'
    if poetry run python -m silnlp.nmt.analyze_project_pairs {experiment_name}; then
        echo 'SUCCESS' > $STATUS_FILE
    else
        echo 'FAILED' > $STATUS_FILE
    fi
    rm -f $PID_FILE
"

# Give the screen session a moment to start and create the PID file
sleep 5

# Wait for completion by monitoring the PID file and status file
echo "Waiting for alignment to complete..."
while [ -f "$PID_FILE" ]; do
    sleep 30
done

# Check the final status
if [ -f "$STATUS_FILE" ] && [ "$(cat $STATUS_FILE)" = "SUCCESS" ]; then
    echo "Alignment task completed successfully"
    rm -f "$STATUS_FILE"
    exit 0
else
    echo "Alignment task failed"
    rm -f "$STATUS_FILE"
    exit 1
fi
"""

    def _generate_train_script(
        self,
        experiment_name: str,
    ) -> str:
        """Generate script content for training task"""

        return f"""
# Training task
echo "Starting training task..."
echo "Experiment name: {experiment_name}"

# TODO: Implement actual training logic here
cd {SILNLP_ROOT}

# Create a unique session name and files for this task
SESSION_NAME="train_{self.task.id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"

echo "Running training in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
CUDA_VISIBLE_DEVICES={CUDA_DEVICE} screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting alignment process...'
    if poetry run python -m silnlp.nmt.experiment --clearml-queue local {experiment_name}; then
        echo 'SUCCESS' > $STATUS_FILE
    else
        echo 'FAILED' > $STATUS_FILE
    fi
    rm -f $PID_FILE
"

# Give the screen session a moment to start and create the PID file
sleep 5

# Wait for completion by monitoring the PID file and status file
echo "Waiting for training to complete..."
while [ -f "$PID_FILE" ]; do
    sleep 30
done

# Check the final status
if [ -f "$STATUS_FILE" ] && [ "$(cat $STATUS_FILE)" = "SUCCESS" ]; then
    echo "Training task completed successfully"
    rm -f "$STATUS_FILE"
    exit 0
else
    echo "Training task failed"
    rm -f "$STATUS_FILE"
    exit 1
fi
"""

    def _run_script(self, script_content: str, task_type: str) -> bool:
        """Run a shell script for the given task type"""
        try:
            # Create a temporary script file
            script_filename = f"task_{self.task.id}_{task_type}.sh"
            script_path = f"/tmp/{script_filename}"

            # Write script to file
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")  # Exit on any error
                f.write(f"# Task: {self.task.id} - Kind: {task_type}\n")
                f.write(script_content)

            # Make script executable
            os.chmod(script_path, 0o755)

            # Execute script
            self.logger.info(f"Running script: {script_path}")
            result = subprocess.run(
                ["/bin/bash", script_path],
                capture_output=True,
                text=True,
                timeout=86400,  # 24 hour timeout
            )

            # Log output
            if result.stdout:
                self.logger.info(f"Script output: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Script stderr: {result.stderr}")

            # Clean up
            try:
                os.remove(script_path)
            except:
                pass

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            self.logger.error(f"Script execution timed out for task {self.task.id}")
            return False
        except Exception as e:
            self.logger.error(f"Error running script for task {self.task.id}: {e}")
            return False


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


def main():
    """Main entry point"""
    try:
        worker = SilAutoWorker()
        worker.run()
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        exit(1)


if __name__ == "__main__":
    main()
