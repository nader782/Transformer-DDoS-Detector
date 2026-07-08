"""
src/detector.py
---------------
DDoSDetector: the central orchestrator that ties together the capture,
inference, logging, and dashboard state-push pipeline.

It runs ``LiveSniffer`` in the calling context (either via the start/stop
API or inline) and for every completed flow:

  1. Calls ``FlowPredictor.predict(flow_data)``
  2. If attack or warning → calls ``AttackLogger.log(...)``
  3. Pushes a payload to the Flask dashboard via HTTP POST to /api/update

Data sent to the dashboard (POST /api/update):
-----------------------------------------------
{
  "last_score":       <float>,          # max softmax probability
  "is_attack":        <bool>,
  "attack_type":      <str>,            # predicted label
  "last_features":    {<20 features>},  # selected features values
  "traffic_history":  [{timestamp, packets, bytes}],  # single-entry list
  "alerts":           [{timestamp, event_type, score, src_ip, dst_ip, protocol}]
}
"""

from __future__ import annotations
from src.logger import AttackLogger
from src.inference.predictor import FlowPredictor
from src.capture.sniffer import LiveSniffer

import sys
import time
import threading
from pathlib import Path
from typing import Callable

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper: map cicflowmeter column names → dashboard-friendly names
# ---------------------------------------------------------------------------

# cicflowmeter uses camelCase / space-separated names in some versions.
# We normalise the keys by stripping spaces and lower-casing.
def _normalise_keys(data: dict) -> dict:
    """Return a new dict with normalised (lower, underscore) keys."""
    normalised = {}
    for k, v in data.items():
        # 'Src IP' → 'src_ip', 'Flow Duration' → 'flow_duration', etc.
        new_key = (
            k.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        normalised[new_key] = v
    return normalised


class DDoSDetector:
    """Orchestrates capture → inference → logging → dashboard push.

    Parameters
    ----------
    interface : str
        Network interface to capture on.
    model_dir : str | Path
        Directory with trained model artifacts.
    log_dir : str | Path
        Directory to write attack log files.
    dashboard_url : str
        Base URL of the Flask dashboard (e.g. ``"http://localhost:5000"``).
    attack_threshold : float
        Prediction score above which traffic is classified as confirmed attack.
    warning_threshold : float
        Score between this and ``attack_threshold`` triggers a warning.
    verbose : bool
        Enable verbose cicflowmeter output.
    """

    def __init__(
        self,
        interface: str,
        model_dir: str | Path = "model",
        log_dir: str | Path = "logs",
        dashboard_url: str = "http://localhost:5000",
        attack_threshold: float = 0.75,
        warning_threshold: float = 0.50,
        verbose: bool = False,
    ) -> None:
        self._interface = interface
        self._dashboard_url = dashboard_url.rstrip("/")
        self._attack_threshold = attack_threshold
        self._warning_threshold = warning_threshold

        # Initialise components
        self._predictor = FlowPredictor(model_dir=model_dir)
        self._attack_logger = AttackLogger(log_dir=log_dir)
        self._sniffer = LiveSniffer(
            interface=interface,
            callback=self._on_flow,
            verbose=verbose,
        )

        # HTTP session for posting to dashboard (reused for efficiency)
        self._http = requests.Session()
        self._stopped = threading.Event()

    # ------------------------------------------------------------------
    # Flow callback (called from the sniffer thread)
    # ------------------------------------------------------------------

    def _on_flow(self, raw_flow: dict) -> None:
        """Process a single completed flow end-to-end."""
        # Normalise keys so feature lookup is consistent
        flow = _normalise_keys(raw_flow)

        # --- Inference ---
        try:
            result = self._predictor.predict(flow)
        except Exception as exc:
            print(f"[DDoSDetector] Prediction error: {exc}")
            return

        label = result["label"]
        score = result["score"]
        is_attack = result["is_attack"]
        is_warning = result["is_warning"]

        # --- Logging (attack and warning events) ---
        if is_attack or is_warning:
            try:
                self._attack_logger.log(flow, result)
            except Exception as exc:
                print(f"[DDoSDetector] Logger error: {exc}")

        # --- Dashboard push ---
        self._push_to_dashboard(flow, result)

    # ------------------------------------------------------------------
    # Dashboard state push
    # ------------------------------------------------------------------

    def _push_to_dashboard(self, flow: dict, result: dict) -> None:
        """POST flow data and prediction result to the Flask dashboard."""
        label = result["label"]
        score = result["score"]
        is_attack = result["is_attack"]
        is_warning = result["is_warning"]
        now = time.time()

        # Extract selected features for the feature panel
        selected_features = {
            feat: float(flow.get(feat, 0) or 0)
            for feat in self._predictor.selected_features
        }

        # Traffic history entry (packets and bytes per flow)
        traffic_entry = {
            "timestamp": now,
            "packets": float(flow.get("tot_fwd_pkts", 0) or 0)
            + float(flow.get("tot_bwd_pkts", 0) or 0),
            "bytes": float(flow.get("totlen_fwd_pkts", 0) or 0)
            + float(flow.get("totlen_bwd_pkts", 0) or 0),
        }

        # Alert entry (only for attack / warning)
        alerts: list = []
        if is_attack or is_warning:
            event_type = f"Warning [{label}]" if is_warning else label
            alerts.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "event_type": event_type,
                "score": round(score, 6),
                "src_ip": flow.get("src_ip", ""),
                "dst_ip": flow.get("dst_ip", ""),
                "protocol": int(flow.get("protocol", 0) or 0),
            })

        payload = {
            "last_score": score,
            "is_attack": is_attack,
            "attack_type": label,
            "last_features": selected_features,
            "traffic_history": [traffic_entry],
            "alerts": alerts,
        }

        try:
            resp = self._http.post(
                f"{self._dashboard_url}/api/update",
                json=payload,
                timeout=2,
            )
            resp.raise_for_status()
        except Exception:
            # Never let a dashboard push error crash the sniffer thread.
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the sniffer (non-blocking — runs in scapy's internal threads)."""
        self._sniffer.start()

    def stop(self) -> None:
        """Stop capture, flush remaining flows, and close the HTTP session."""
        self._sniffer.stop()
        self._stopped.set()
        self._http.close()
        print("[DDoSDetector] Stopped.")
