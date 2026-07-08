"""
src/logger.py
-------------
AttackLogger: writes structured DDoS attack / warning events to a rotating
log file using Python's built-in logging module.

Each log line is a JSON-serialisable record containing:
  timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
  predicted_label, score, flow_duration, tot_fwd_pkts, tot_bwd_pkts,
  all_scores (dict of class → probability)

The log directory is created automatically on first use.
Max file size: 10 MB — keeps 5 rotated backups (50 MB total).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


_LOG_DIR = "logs"
_LOG_FILE = "ddos_detections.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


class AttackLogger:
    """Rotating-file logger for DDoS detection events.

    Parameters
    ----------
    log_dir : str | Path
        Directory where the log file is written. Created if it does not exist.
    log_file : str
        Log file name inside ``log_dir``.
    """

    def __init__(
        self,
        log_dir: str | Path = _LOG_DIR,
        log_file: str = _LOG_FILE,
    ) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file

        self._logger = logging.getLogger(f"ddos_detect.{log_path}")
        self._logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers if logger is re-used
        if not self._logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

        print(f"[AttackLogger] Logging to: {log_path.resolve()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, flow_data: dict, prediction: dict) -> None:
        """Write a structured event record for an attack or warning.

        Parameters
        ----------
        flow_data : dict
            Raw cicflowmeter flow dict (used to extract IP / port / protocol).
        prediction : dict
            Result from ``FlowPredictor.predict()``, must contain at least:
            ``label``, ``score``, ``is_attack``, ``is_warning``, ``all_scores``.
        """
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": flow_data.get("src_ip", ""),
            "dst_ip": flow_data.get("dst_ip", ""),
            "src_port": int(flow_data.get("src_port", 0) or 0),
            "dst_port": int(flow_data.get("dst_port", 0) or 0),
            "protocol": int(flow_data.get("protocol", 0) or 0),
            "predicted_label": prediction.get("label", "UNKNOWN"),
            "score": round(float(prediction.get("score", 0.0)), 6),
            "is_attack": prediction.get("is_attack", False),
            "is_warning": prediction.get("is_warning", False),
            "flow_duration": flow_data.get("flow_duration", 0),
            "tot_fwd_pkts": flow_data.get("tot_fwd_pkts", 0),
            "tot_bwd_pkts": flow_data.get("tot_bwd_pkts", 0),
            "all_scores": prediction.get("all_scores", {}),
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))
