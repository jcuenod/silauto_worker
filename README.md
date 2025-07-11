# SILAUTO Worker

A worker service that processes tasks for the SILAUTO system. Handles four types of tasks: Translate, Extract, Align, and Train.

## Installation

**Note:** This project requires SSH access to the host machine. All commands are executed via SSH, whether running locally or inside Docker. This setup allows the worker to interact with the host environment, which is necessary for supporting Docker-based workflows.

### SSH Setup (Required for All Installations)

1. Generate an SSH key for container/worker access:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/container_key -N ""
   cat ~/.ssh/container_key.pub >> ~/.ssh/authorized_keys
   ```
2. Ensure the SSH host service is running on your machine.

### Local Installation

1. Clone the repository and navigate to the worker directory.
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

## Environment Variables

| Variable        | Description                                                       | Example                 |
| --------------- | ----------------------------------------------------------------- | ----------------------- |
| `SILAUTO_URL`   | Base URL of the SILAUTO API server                                | `http://localhost:8000` |
| `SILNLP_ROOT`   | Path to the SILNLP root directory                                 | `~/silnlp`              |
| `SILNLP_DATA`   | Path to the silnlp_data directory                                 | `~/silnlp_data/`        |
| `CUDA_DEVICE`   | GPU device index to use                                           | `0`                     |
| `USFM2PDF_PATH` | Path to the [usfm2pdf](https://github.com/jcuenod/usfm2pdf) tool. | `/home/user/usfm2pdf`   |

## Volumes

To enable SSH access from within the container, mount your SSH private key as a volume:

```bash
-v /path/to/your/id_ed25519:/app/.ssh/id_ed25519:ro
```

This ensures the worker can authenticate via SSH when running inside Docker. Replace `/path/to/your/id_ed25519` with the path to your SSH private key.

## License

This project is part of the SILAUTO system.
