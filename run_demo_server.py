#!/usr/bin/env python3
"""
AEZ Evolution — Server Launcher (minimal)

Same as demo.py but without the banner. For scripts/CI.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from engine.server import app

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"AEZ Evolution server: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
