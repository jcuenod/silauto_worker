import os

SILAUTO_URL = os.getenv('SILAUTO_URL', "")
if not SILAUTO_URL:
    raise ValueError("SILAUTO_URL environment variable is required")    

SILNLP_ROOT = os.getenv('SILNLP_ROOT', "")
if not SILNLP_ROOT:
    raise ValueError("SILNLP_ROOT environment variable is required")

SILNLP_EXPERIMENTS_ROOT = os.getenv('SILNLP_EXPERIMENTS_ROOT', "")
if not SILNLP_EXPERIMENTS_ROOT:
    raise ValueError("SILNLP_EXPERIMENTS_ROOT environment variable is required")

CUDA_DEVICE = os.getenv('CUDA_DEVICE', "")
if not CUDA_DEVICE:
    raise ValueError("CUDA_DEVICE environment variable is required")

USFM2PDF_PATH = os.getenv('USFM2PDF_PATH', None)
