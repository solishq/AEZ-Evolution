#!/usr/bin/env python3
"""
AEZ Evolution — Demo Launcher

Copyright (c) 2026 SolisHQ (github.com/solishq). MIT License.

One command. Full god-mode dashboard.
Neural agents. Trust cascades. Adversarial attacks. Real-time visualization.

Usage:
    python demo.py              # Start on port 8000
    python demo.py --port 3000  # Custom port
    python demo.py --no-browser # Don't auto-open browser
"""

import sys
import os
import argparse
import threading
import time
import webbrowser

# Ensure engine is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          ▄▀▄ ▄▀▀ ▀▀█  EVOLUTION                                ║
║          █▀█ ██▄  ▄▀   Neural Trust Networks                   ║
║          ▀ ▀ ▀▀▀ ▀▀▀   God-Mode Playground                     ║
║                                                                  ║
║  Neural agents discover cooperation through evolution.           ║
║  4D trust tensors. Cascade collapse. Adversarial injection.      ║
║  No hardcoded strategies. No assumptions. Pure emergence.        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def preflight():
    """Verify all dependencies before starting."""
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        import fastapi
    except ImportError:
        missing.append('fastapi')
    try:
        import uvicorn
    except ImportError:
        missing.append('uvicorn')
    try:
        import pydantic
    except ImportError:
        missing.append('pydantic')

    if missing:
        print(f"\n  Missing dependencies: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    # Verify engine imports
    try:
        from engine.agent import NeuralAgent
        from engine.trust import TrustNetwork, TrustState
        from engine.evolution import Evolution, Attacks
        from engine.narrator import Narrator
        from engine.server import app
    except ImportError as e:
        print(f"\n  Engine import failed: {e}")
        print(f"  Run this from the aez-evolution directory.")
        sys.exit(1)

    return True


def open_browser(port, delay=1.5):
    """Open dashboard in browser after server starts."""
    time.sleep(delay)
    url = f"http://localhost:{port}"
    print(f"\n  Opening {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description='AEZ Evolution — God-Mode Demo')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t auto-open browser')
    args = parser.parse_args()

    print(BANNER)

    print("  Preflight check...")
    preflight()
    print("  All systems go.\n")

    print(f"  Server:    http://localhost:{args.port}")
    print(f"  Dashboard: http://localhost:{args.port}")
    print(f"  API docs:  http://localhost:{args.port}/docs")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print()
    print("  Controls:")
    print("    [New Simulation]  Create 50 neural agents")
    print("    [Step]            Run one round of interactions")
    print("    [Auto]            Toggle auto-run (2 rounds/sec)")
    print("    [+10] [+50]       Batch rounds")
    print("    [Select]          Natural selection (kill weak, reproduce strong)")
    print("    [Sybil Attack]    Inject colluding agents (auto-detected!)")
    print("    [Trojan Attack]   Plant sleeper agents that betray later")
    print("    [Eclipse Attack]  Isolate a target agent")
    print("    [Economic Sliders] Change payoff matrix in real-time")
    print()
    print("  Sybil Detection:")
    print("    The trust network automatically detects colluding agents")
    print("    using behavioral inversion analysis. Sybils glow purple")
    print("    when caught. Watch them get isolated and eliminated.")
    print()
    print("  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    import uvicorn
    from engine.server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Simulation terminated. The agents will remember.\n")
        sys.exit(0)
