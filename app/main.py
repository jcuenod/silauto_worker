#!/usr/bin/env python3
"""
SILAUTO Worker - Handles Translate, Extract, Align, and Train tasks
"""

import os
import time
import json
import logging
import subprocess
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    TRANSLATE = "translate"
    EXTRACT = "extract"
    ALIGN = "align"
    TRAIN = "train"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    type: TaskType
    data: Dict[str, Any]
    script_path: Optional[str] = None


class GPUChecker:
    """Check GPU availability using nvidia-smi and torch"""
    
    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU is available using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
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
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpus.append({
                                'name': parts[0],
                                'memory_total': int(parts[1]),
                                'memory_used': int(parts[2]),
                                'memory_free': int(parts[3]),
                                'utilization': int(parts[4]),
                                'temperature': int(parts[5])
                            })
                return {'gpus': gpus, 'count': len(gpus)}
            
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return {'gpus': [], 'count': 0}


class TaskExecutor:
    """Execute different types of tasks"""
    
    def __init__(self, task: Task):
        self.task = task
        self.logger = logging.getLogger(f"{__name__}.TaskExecutor")
    
    def execute(self) -> bool:
        """Execute the task based on its type"""
        try:
            self.logger.info(f"Executing {self.task.type.value} task {self.task.id}")
            
            if self.task.type == TaskType.TRANSLATE:
                return self._execute_translate()
            elif self.task.type == TaskType.EXTRACT:
                return self._execute_extract()
            elif self.task.type == TaskType.ALIGN:
                return self._execute_align()
            elif self.task.type == TaskType.TRAIN:
                return self._execute_train()
            else:
                self.logger.error(f"Unknown task type: {self.task.type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing task {self.task.id}: {e}")
            return False
    
    def _execute_translate(self) -> bool:
        """Execute translation task"""
        script_content = self.task.data.get('script', '')
        if not script_content:
            self.logger.error("No script content provided for translate task")
            return False
        
        return self._run_script(script_content, "translate")
    
    def _execute_extract(self) -> bool:
        """Execute extraction task"""
        script_content = self.task.data.get('script', '')
        if not script_content:
            self.logger.error("No script content provided for extract task")
            return False
        
        return self._run_script(script_content, "extract")
    
    def _execute_align(self) -> bool:
        """Execute alignment task"""
        script_content = self.task.data.get('script', '')
        if not script_content:
            self.logger.error("No script content provided for align task")
            return False
        
        return self._run_script(script_content, "align")
    
    def _execute_train(self) -> bool:
        """Execute training task"""
        script_content = self.task.data.get('script', '')
        if not script_content:
            self.logger.error("No script content provided for train task")
            return False
        
        return self._run_script(script_content, "train")
    
    def _run_script(self, script_content: str, task_type: str) -> bool:
        """Run a shell script for the given task type"""
        try:
            # Create a temporary script file
            script_filename = f"task_{self.task.id}_{task_type}.sh"
            script_path = f"/tmp/{script_filename}"
            
            # Write script to file
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")  # Exit on any error
                f.write(f"# Task: {self.task.id} - Type: {task_type}\n")
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Execute script
            self.logger.info(f"Running script: {script_path}")
            result = subprocess.run(
                ['/bin/bash', script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
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
        self.base_url = os.getenv('SILAUTO_URL')
        if not self.base_url:
            raise ValueError("SILAUTO_URL environment variable is required")
        
        self.base_url = self.base_url.rstrip('/')
        self.logger = logging.getLogger(f"{__name__}.SilAutoWorker")
        self.session = requests.Session()
        
        self.logger.info("Worker initialized and ready to process tasks")
    
    def fetch_next_task(self) -> Optional[Task]:
        """Fetch the next available task from the API"""
        try:
            # Check GPU availability before requesting tasks
            gpu_available = GPUChecker.check_gpu_available()
            gpu_info = GPUChecker.get_gpu_info()
            
            # Include GPU status in the request to help server decide which tasks to assign
            url = f"{self.base_url}/tasks/next"
            self.logger.debug(f"Fetching next task from: {url} (GPU available: {gpu_available})")
            
            # Add GPU info to request headers/params so server can make informed decisions
            headers = {
                'X-GPU-Available': str(gpu_available).lower(),
                'X-GPU-Count': str(gpu_info.get('count', 0))
            }
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 404:
                self.logger.debug("No tasks available (404)")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Parse task data
            task_type = TaskType(data['type'])
            task = Task(
                id=data['id'],
                type=task_type,
                data=data.get('data', {})
            )
            
            self.logger.info(f"Fetched task: {task.id} (type: {task.type.value}) - GPU available: {gpu_available}")
            return task
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching next task: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing task data: {e}")
            return None
    
    def update_task_status(self, task_id: str, status: TaskStatus, message: str = "") -> bool:
        """Update the status of a task"""
        try:
            url = f"{self.base_url}/tasks/{task_id}"
            
            # Include current GPU status in the update
            gpu_info = GPUChecker.get_gpu_info()
            payload = {
                'status': status.value,
                'message': message,
                'timestamp': time.time(),
                'gpu_available': gpu_info.get('count', 0) > 0,
                'gpu_count': gpu_info.get('count', 0)
            }
            
            self.logger.debug(f"Updating task {task_id} status to {status.value}")
            
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            self.logger.info(f"Successfully updated task {task_id} status to {status.value}")
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