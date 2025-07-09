# SILAUTO Worker

A worker service that processes tasks for the SILAUTO system. Handles four types of tasks: Translate, Extract, Align, and Train.

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
   # Edit .env with your env vars
   ```

4. Run the worker:
   ```bash
   python -m app.main
   ```

### Docker Installation

1. Build the Docker image:

   ```bash
   docker build -t silauto-worker .
   ```

## Configuration

### Environment Variables

| Variable        | Description                                                       | Example                 |
|-----------------|-------------------------------------------------------------------|-------------------------|
| `SILAUTO_URL`   | Base URL of the SILAUTO API server                                | `http://localhost:8000` |
| `SILNLP_ROOT`   | Path to the SILNLP root directory                                 | `~/silnlp`              |
| `SILNLP_DATA`   | Path to the silnlp_data directory                                 | `~/silnlp_data/`        |
| `CUDA_DEVICE`   | GPU device index to use                                           | `0`                     |
| `USFM2PDF_PATH` | Path to the [usfm2pdf](https://github.com/jcuenod/usfm2pdf) tool. | `/home/user/usfm2pdf`   |

## GPU Support

The worker automatically detects GPU availability using `nvidia-smi`. GPU information is logged on startup and can be used by task scripts.

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

## License

This project is part of the SILAUTO system.
