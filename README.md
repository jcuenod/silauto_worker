# SILAUTO Worker

A worker service that processes tasks for the SILAUTO system. Handles four types of tasks: Translate, Extract, Align, and Train.

## Features

- **GPU Support**: Automatically detects and utilizes available GPUs using nvidia-smi
- **Task Types**: Supports Translate, Extract, Align, and Train tasks
- **Fault Tolerance**: Handles API errors gracefully with retry logic
- **Logging**: Comprehensive logging for monitoring and debugging
- **Containerized**: Ready-to-use Docker configuration

## Requirements

- Python 3.8+
- NVIDIA GPU (optional, but recommended for ML tasks)
- NVIDIA drivers and nvidia-smi (for GPU support)
- Docker (for containerized deployment)

## Installation

### Local Installation

1. Clone the repository and navigate to the worker directory
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your SILAUTO_URL
   ```

4. Run the worker:
   ```bash
   ./start.sh
   ```

### Docker Installation

1. Build the Docker image:

   ```bash
   docker build -t silauto-worker .
   ```

2. Run with Docker Compose:
   ```bash
   # Set SILAUTO_URL environment variable first
   export SILAUTO_URL=http://your-api-server:8000
   docker compose up -d
   ```

## Configuration

### Environment Variables

- `SILAUTO_URL` (required): Base URL of the SILAUTO API server
- `WORKER_ID` (optional): Unique identifier for this worker instance
- `WORKER_NAME` (optional): Human-readable name for this worker
- `LOG_LEVEL` (optional): Logging level (DEBUG, INFO, WARNING, ERROR)

### Task Format

The worker expects tasks in the following format from the API:

```json
{
  "id": "task-123",
  "type": "translate|extract|align|train",
  "data": {
    "script": "#!/bin/bash\necho 'Task script content'"
  }
}
```

## API Endpoints

The worker interacts with these API endpoints:

- `GET /tasks/next`: Fetch the next available task
- `POST /tasks/{task_id}`: Update task status

## Task Types

### Translate

Handles translation tasks by executing the provided shell script.

### Extract

Handles data extraction tasks by executing the provided shell script.

### Align

Handles alignment tasks by executing the provided shell script.

### Train

Handles training tasks by executing the provided shell script.

## GPU Support

The worker automatically detects GPU availability using `nvidia-smi`. GPU information is logged on startup and can be used by task scripts.

## Logging

Logs are written to stdout with the following format:

```
2025-06-25 10:30:45,123 - main.SilAutoWorker - INFO - Worker initialized. GPU available: True
```

## Error Handling

- **Network errors**: Retries with exponential backoff
- **Task failures**: Marked as FAILED with error details
- **GPU errors**: Gracefully degrades to CPU-only mode
- **Script timeouts**: Tasks timeout after 1 hour

## Development

### Running Locally

```bash
# Set environment variable
export SILAUTO_URL=http://localhost:8000

# Run directly
python app/main.py

# Or use the startup script
./start.sh
```

### Testing

```bash
# Test GPU detection
python -c "from app.main import GPUChecker; print(GPUChecker.check_gpu_available())"

# Test API connectivity
curl $SILAUTO_URL/tasks/next
```

## Deployment

### Production Deployment

1. Use the provided Dockerfile for containerized deployment
2. Set up monitoring and log aggregation
3. Configure resource limits appropriate for your workload
4. Use orchestration tools like Kubernetes for scaling

### Scaling

Multiple worker instances can run simultaneously. Each worker polls for tasks independently, providing horizontal scaling capabilities.

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA drivers and nvidia-smi are installed
2. **API connection errors**: Check SILAUTO_URL and network connectivity
3. **Task script failures**: Check task script syntax and permissions
4. **Memory issues**: Monitor GPU memory usage for large tasks

### Debug Mode

Set `LOG_LEVEL=DEBUG` for verbose logging:

```bash
export LOG_LEVEL=DEBUG
./start.sh
```

## License

This project is part of the SILAUTO system.
