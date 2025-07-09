#!/usr/bin/env python3
"""
SILAUTO Worker - Handles Draft (Translate), Extract, Align, and Train tasks
"""

from app.worker import SilAutoWorker
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    try:
        worker = SilAutoWorker()
        worker.run()
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        exit(1)


if __name__ == "__main__":
    main()
