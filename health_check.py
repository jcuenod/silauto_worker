#!/usr/bin/env python3
"""
Health check script for SILAUTO Worker
"""

import os
import sys
import requests
from app.main import GPUChecker

def check_api_connectivity():
    """Check if the API server is reachable"""
    try:
        base_url = os.getenv('SILAUTO_URL')
        if not base_url:
            return False, "SILAUTO_URL not set"
        
        response = requests.get(f"{base_url.rstrip('/')}/tasks/next", timeout=10)
        if response.status_code in [200, 404]:  # Both are valid responses
            return True, "API connectivity OK"
        else:
            return False, f"API returned status {response.status_code}"
    
    except Exception as e:
        return False, f"API connectivity error: {e}"

def check_gpu():
    """Check GPU availability"""
    try:
        gpu_available = GPUChecker.check_gpu_available()
        gpu_info = GPUChecker.get_gpu_info()
        
        if gpu_available:
            return True, f"GPU available: {gpu_info['count']} GPU(s) detected"
        else:
            return True, "No GPU detected (CPU-only mode)"
    
    except Exception as e:
        return False, f"GPU check error: {e}"

def main():
    """Run health checks"""
    print("SILAUTO Worker Health Check")
    print("=" * 40)
    
    all_good = True
    
    # Check API connectivity
    api_ok, api_msg = check_api_connectivity()
    print(f"API Connectivity: {'✓' if api_ok else '✗'} {api_msg}")
    if not api_ok:
        all_good = False
    
    # Check GPU
    gpu_ok, gpu_msg = check_gpu()
    print(f"GPU Status: {'✓' if gpu_ok else '✗'} {gpu_msg}")
    if not gpu_ok:
        all_good = False
    
    # Check Python dependencies
    try:
        import requests
        import torch
        print("Dependencies: ✓ All required packages available")
    except ImportError as e:
        print(f"Dependencies: ✗ Missing package: {e}")
        all_good = False
    
    print("=" * 40)
    if all_good:
        print("Health Check: ✓ All systems ready")
        sys.exit(0)
    else:
        print("Health Check: ✗ Some issues detected")
        sys.exit(1)

if __name__ == "__main__":
    main()
