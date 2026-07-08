"""
main.py
-------
Entry point for the Real-Time DDoS Detection System.

Usage
-----
  # Capture live traffic on a network interface:
  python main.py --interface "Wi-Fi"

  # With custom options:
  python main.py --interface "Ethernet" --port 5000 --log-dir logs/ --threshold 0.75

  # List available interfaces (Windows):
  python -c "from scapy.arch import get_if_list; print(get_if_list())"

Notes
-----
- Requires Administrator / root privileges for raw socket capture.
- On Windows, use the full Npcap interface name (e.g. "Wi-Fi", "Ethernet").
- The Flask dashboard is available at http://localhost:<port> while running.
"""

from __future__ import annotations
from src.dashboard import create_app
from src.detector import DDoSDetector

import argparse
import sys
import signal
import threading
from pathlib import Path

# Ensure project root is on the Python path regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time DDoS Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interface", "-i",
        required=True,
        help='Network interface to capture on (e.g. "Wi-Fi", "eth0")',
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Flask dashboard host",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Flask dashboard port",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for attack log files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Score threshold above which traffic is classified as attack (0–1)",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=0.50,
        help="Score threshold above which traffic triggers a warning (0–1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose cicflowmeter output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_url = f"http://127.0.0.1:{args.port}"

    print("=" * 60)
    print("  SIEM — Real-Time DDoS Detection System")
    print("=" * 60)
    print(f"  Interface  : {args.interface}")
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Log dir    : {args.log_dir}")
    print(f"  Dashboard  : http://{args.host}:{args.port}")
    print(
        f"  Thresholds : attack={args.threshold}, warning={args.warning_threshold}")
    print("=" * 60)
    print()

    # ---------------------------------------------------------------
    # 1. Initialise the detector (loads the model once)
    # ---------------------------------------------------------------
    detector = DDoSDetector(
        interface=args.interface,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        dashboard_url=dashboard_url,
        attack_threshold=args.threshold,
        warning_threshold=args.warning_threshold,
        verbose=args.verbose,
    )

    # ---------------------------------------------------------------
    # 2. Start the detector in a background daemon thread
    # ---------------------------------------------------------------
    detector_thread = threading.Thread(
        target=detector.start,
        name="ddos-detector",
        daemon=True,
    )
    detector_thread.start()

    # ---------------------------------------------------------------
    # 3. Graceful shutdown on SIGINT / SIGTERM
    # ---------------------------------------------------------------
    def _shutdown(signum, frame):
        print("\n[main] Shutdown signal received — stopping capture…")
        detector.stop()
        print("[main] Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (OSError, AttributeError):
        pass  # SIGTERM not available on all platforms (e.g. Windows)

    # ---------------------------------------------------------------
    # 4. Start Flask dashboard in the main thread (blocks)
    # ---------------------------------------------------------------
    flask_app = create_app()
    print(f"[main] Dashboard starting at http://{args.host}:{args.port}")
    flask_app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,  # avoid double-start with signal handlers
    )


if __name__ == "__main__":
    main()
