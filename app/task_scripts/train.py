from app.env import CUDA_DEVICE, SILNLP_ROOT


def generate_train_script(
    task_id: str,
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
SESSION_NAME="train_{task_id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"

echo "Running training in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
CUDA_VISIBLE_DEVICES={CUDA_DEVICE} screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting training process...'
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
