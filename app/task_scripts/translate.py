from typing import List
from app.env import SILNLP_EXPERIMENTS_ROOT, SILNLP_ROOT, USFM2PDF_PATH


def generate_translate_script(
    task_id: str,
    train_task_id: str,
    source_project_id: str,
    book_names: List[str],
    source_script_code: str,
    target_script_code: str,
    experiment_name: str,
) -> str:
    """Generate script content for translation task"""
    books_str = ",".join(book_names)

    usfm_generate_script = (
        f"""
        # Try to generate PDFs for the drafts, but ignore any failure
        cd $USFM2PDF_PATH
        python main.py -n --header 'AI Draft' '{SILNLP_EXPERIMENTS_ROOT}{experiment_name}/infer/*/{source_project_id}/*.SFM'"""
        if USFM2PDF_PATH
        else ""
    )

    return f"""
# Translate task for (train) project: {train_task_id}
echo "Starting translation task..."
echo "Train task ID: {train_task_id}"
echo "Experiment name: {experiment_name}"
echo "Source project ID: {source_project_id}"
echo "Books to translate: {books_str}"
echo "Source script code: {source_script_code}"
echo "Target script code: {target_script_code}"

cd {SILNLP_ROOT}

# Create a unique session name and files for this task
SESSION_NAME="translate_{task_id}"
PID_FILE="/tmp/$SESSION_NAME.pid"
STATUS_FILE="/tmp/$SESSION_NAME.status"
LOG_FILE="/tmp/$SESSION_NAME.log"
USFM2PDF_PATH={USFM2PDF_PATH if USFM2PDF_PATH else ""}

echo "Running translate in screen session: $SESSION_NAME"
echo "Output will be logged to: $LOG_FILE"
# Start screen session with a wrapper that tracks completion
screen -L -d -m -S "$SESSION_NAME" bash -c "
    echo $$ > $PID_FILE
    exec > >(tee -a $LOG_FILE) 2>&1
    echo 'Starting translation process...'
    if poetry run python -m silnlp.nmt.translate {experiment_name} --src-project {source_project_id} --books {books_str} --src-iso {source_script_code} --trg-iso {target_script_code} --checkpoint best; then
        {usfm_generate_script}
        echo 'SUCCESS' > $STATUS_FILE
    else
        echo 'FAILED' > $STATUS_FILE
    fi
    rm -f $PID_FILE
"

# Give the screen session a moment to start and create the PID file
sleep 5

# Wait for completion by monitoring the PID file and status file
echo "Waiting for translation to complete..."
while [ -f "$PID_FILE" ]; do
    sleep 30
done

# Check the final status
if [ -f "$STATUS_FILE" ] && [ "$(cat $STATUS_FILE)" = "SUCCESS" ]; then
    echo "Translation task completed successfully"
    rm -f "$STATUS_FILE"
    exit 0
else
    echo "Translation task failed"
    rm -f "$STATUS_FILE"
    exit 1
fi
"""
