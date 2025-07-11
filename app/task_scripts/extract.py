from app.env import SILNLP_ROOT
from app.models import TaskStatus


def generate_extract_script(task_id: str, project_id: str) -> str:
    """Generate script content for extraction task"""
    return f"""
# Extraction task for project: {project_id}
echo "Starting extraction task..."
echo "Project ID: {project_id}"

cd {SILNLP_ROOT}

# Create a unique session name and files for this task
SESSION_NAME="extract_{task_id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"

echo "Running extraction in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting extraction process...'
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
    # Check if the screen session is still running
    if ! screen -list | grep -q "$SESSION_NAME"; then
        echo "Screen session $SESSION_NAME is no longer running!"
        echo "Likely crash for pid: $PID_FILE"
        break
    fi
    sleep 30
done

# Check the final status
if [ -f "$STATUS_FILE" ] && [ "$(cat $STATUS_FILE)" = "SUCCESS" ]; then
    echo "__STATUS:{TaskStatus.COMPLETED}__"
    echo "Extraction task completed successfully"
    rm -f "$STATUS_FILE"
else
    echo "__STATUS:{TaskStatus.FAILED}__"
    echo "Extraction task failed"
    rm -f "$STATUS_FILE"
fi
"""
