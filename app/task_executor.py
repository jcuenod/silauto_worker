import logging
import os
import subprocess
from app.task_scripts.translate import generate_translate_script
from app.task_scripts.extract import generate_extract_script
from app.task_scripts.align import generate_align_script
from app.task_scripts.train import generate_train_script
from app.models import Task, TaskKind


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
            experiment_name = str(params.get("experiment_name"))
            train_task_id = str(params.get("train_task_id"))
            source_project_id = str(params.get("source_project_id"))
            book_names = list(params.get("book_names", []))
            source_script_code = str(params.get("source_script_code"))
            target_script_code = str(params.get("target_script_code"))

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
            script_content = generate_translate_script(
                self.task.id,
                train_task_id,
                source_project_id,
                book_names,
                source_script_code,
                target_script_code,
                experiment_name,
            )

            return self._run_script(script_content, "translate")

        except Exception as e:
            self.logger.error(f"Error in translate task: {e}")
            return False

    def _execute_extract(self) -> bool:
        """Execute extraction task"""
        try:
            params = self.task.parameters
            project_id = str(params.get("project_id"))

            if not project_id:
                self.logger.error("Missing project_id for extract task")
                return False

            # Generate script content for extraction
            script_content = generate_extract_script(self.task.id, project_id)

            return self._run_script(script_content, "extract")

        except Exception as e:
            self.logger.error(f"Error in extract task: {e}")
            return False

    def _execute_align(self) -> bool:
        """Execute alignment task"""
        try:
            params = self.task.parameters
            experiment_name = str(params.get("experiment_name"))
            target_scripture_file = str(params.get("target_scripture_file"))
            source_scripture_files = list(params.get("source_scripture_files", []))

            if not all(
                [experiment_name, target_scripture_file, source_scripture_files]
            ):
                self.logger.error("Missing required parameters for align task")
                return False

            # Generate script content for alignment
            script_content = generate_align_script(
                self.task.id,
                experiment_name,
                target_scripture_file,
                source_scripture_files,
            )

            return self._run_script(script_content, "align")

        except Exception as e:
            self.logger.error(f"Error in align task: {e}")
            return False

    def _execute_train(self) -> bool:
        """Execute training task"""
        try:
            params = self.task.parameters
            experiment_name = str(params.get("experiment_name"))

            if not experiment_name:
                self.logger.error("Missing required parameters for train task")
                return False

            print(experiment_name)

            # Generate script content for training
            script_content = generate_train_script(
                self.task.id,
                experiment_name,
            )

            return self._run_script(script_content, "train")

        except Exception as e:
            self.logger.error(f"Error in train task: {e}")
            return False

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
                [
                    "ssh",
                    "-i", "/home/worker/.ssh/id_ed25519",
                    "-o", "StrictHostKeyChecking=no",
                    "user@host.docker.internal",
                    "/bin/bash -s",
                ],
                input=open(script_path).read(),
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
