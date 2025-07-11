from typing import List
from app.env import SILNLP_ROOT
from app.models import TaskStatus


def generate_align_script(
    task_id: str,
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
SESSION_NAME="align_{task_id}"
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
    echo "__STATUS:{TaskStatus.COMPLETED}__"
    echo "Alignment task completed successfully"
    rm -f "$STATUS_FILE"
else
    echo "Alignment task failed"
    echo "__STATUS:{TaskStatus.FAILED}__"
    rm -f "$STATUS_FILE"
fi
"""
