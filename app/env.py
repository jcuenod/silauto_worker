import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Required environment variables
SILAUTO_URL = os.getenv('SILAUTO_URL', "")
if not SILAUTO_URL:
    raise ValueError("SILAUTO_URL environment variable is required")    

SILNLP_ROOT = os.getenv('SILNLP_ROOT', "")
if not SILNLP_ROOT:
    raise ValueError("SILNLP_ROOT environment variable is required")

SILNLP_DATA = os.getenv('SILNLP_DATA', "")
if not SILNLP_DATA:
    raise ValueError("SILNLP_DATA environment variable is required")

SILNLP_EXPERIMENTS_ROOT = Path(SILNLP_DATA) / "MT/experiments"

CUDA_DEVICE = os.getenv('CUDA_DEVICE', "")
if not CUDA_DEVICE:
    raise ValueError("CUDA_DEVICE environment variable is required")

# Optional environment variables
USFM2PDF_PATH = os.getenv('USFM2PDF_PATH', None)
if not USFM2PDF_PATH:
    print("USFM2PDF_PATH environment variable is not set, PDF generation will be disabled")