import subprocess
from typing import Any, Dict, Optional
import logging

from .env import CUDA_DEVICE

logger = logging.getLogger(f"{__name__}.SilAutoWorker")


class GPUChecker:
    """Check GPU availability using nvidia-smi and torch"""

    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU is available using nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and lines[0]:
                    logger.info(f"GPU detected: {lines[0]}")
                    return True

            logger.warning("nvidia-smi command failed or no GPU detected")
            return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Error checking GPU: {e}")
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get detailed GPU information"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 6:
                            gpus.append(
                                {
                                    "name": parts[0],
                                    "memory_total": int(parts[1]),
                                    "memory_used": int(parts[2]),
                                    "memory_free": int(parts[3]),
                                    "utilization": int(parts[4]),
                                    "temperature": int(parts[5]),
                                }
                            )

                return {"gpus": gpus, "count": len(gpus)}

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")

        return {"gpus": [], "count": 0}

    @staticmethod
    def get_idle_gpu_id(
        memory_threshold_mb: int = 500, utilization_threshold: int = 10
    ) -> Optional[int]:
        """
        Return the index of an 'idle' GPU (low memory use and low utilization), or None if none found.
        :param memory_threshold_mb: Maximum memory used (in MB) to consider GPU idle.
        :param utilization_threshold: Maximum utilization (%) to consider GPU idle.
        :return: GPU index (int) if idle GPU is found, else None.
        """
        info = GPUChecker.get_gpu_info()
        for idx, gpu in enumerate(info.get("gpus", [])):
            if (
                gpu["memory_used"] <= memory_threshold_mb
                and gpu["utilization"] <= utilization_threshold
            ):
                logger.info(
                    f"GPU {gpu['name']} (id={idx}) is idle (memory_used={gpu['memory_used']}MB, utilization={gpu['utilization']}%)"
                )
                return idx
        logger.info("No idle GPU found")
        return None

    @staticmethod
    def is_gpu_id_idle(
        memory_threshold_mb: int = 500, utilization_threshold: int = 10
    ) -> bool:
        """
        Return the index of an 'idle' GPU (low memory use and low utilization), or None if none found.
        :param memory_threshold_mb: Maximum memory used (in MB) to consider GPU idle.
        :param utilization_threshold: Maximum utilization (%) to consider GPU idle.
        :return: GPU index (int) if idle GPU is found, else None.
        """
        if not CUDA_DEVICE:
            return False

        info = GPUChecker.get_gpu_info()
        try:
            cuda_device_id = int(CUDA_DEVICE)
            gpus = info.get("gpus", [])
            if cuda_device_id < len(gpus):
                gpu = gpus[cuda_device_id]
                return (
                    gpu
                    and gpu["memory_used"] <= memory_threshold_mb
                    and gpu["utilization"] <= utilization_threshold
                )
        except (ValueError, IndexError):
            pass

        return False
