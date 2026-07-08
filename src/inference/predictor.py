"""
src/inference/predictor.py
--------------------------
FlowPredictor: loads the trained TransformerClassifier and its companion
artifacts (scaler, label encoder, selected features list) and exposes a
single ``predict(flow_data)`` method for real-time inference.

Feature pipeline
----------------
1. Extract exactly the 20 selected features from the raw cicflowmeter dict
   (in the exact order specified by selected_features.json).
2. Fill missing features with 0; replace inf / NaN with 0.
3. Scale via the fitted MinMaxScaler (clips out-of-range values).
4. Run through TransformerClassifier in eval mode → softmax probabilities.
5. Return a structured result dict.
"""

from __future__ import annotations

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Make sure the project root is importable when running from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.model_utils import load_model  # noqa: E402


# ---------------------------------------------------------------------------
# Result type (plain dict for easy JSON serialisation)
# ---------------------------------------------------------------------------

# A 'result' dict always has these keys:
#   label      : str   — human-readable predicted class (e.g. "DrDoS_DNS", "BENIGN")
#   score      : float — max class probability (0.0–1.0)
#   is_attack  : bool  — True when predicted class is not "BENIGN"
#   is_warning : bool  — True when score is between WARNING_LOW and ATTACK_THRESH
#                        (uncertain / borderline prediction)
#   all_scores : dict  — {class_name: probability} for all classes

WARNING_LOW = 0.50    # below this → normal
ATTACK_THRESH = 0.75  # above this → confirmed attack, in between → warning


class FlowPredictor:
    """Loads model artifacts once and provides low-latency ``predict()`` calls.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing transformer_ddos.pt, scaler.pkl,
        label_encoder.pkl, and selected_features.json.
    device : torch.device, optional
        Inference device. Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model_dir: str | Path = "model",
        device: torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load everything via existing utility
        self.model, self.scaler, self.label_encoder, self.selected_features = (
            load_model(str(self.model_dir), device=self.device)
        )
        self.model.eval()

        self.class_names: list[str] = list(self.label_encoder.classes_)
        
        # Sequence buffer for inference
        from collections import deque
        self.seq_len = 10
        self.buffer = deque(maxlen=self.seq_len)
        
        print(
            f"[FlowPredictor] Loaded model with {len(self.selected_features)} features "
            f"and {len(self.class_names)} classes on device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, flow_data: dict) -> dict:
        """Run inference on a single cicflowmeter flow dictionary.

        Parameters
        ----------
        flow_data : dict
            Raw flow feature dictionary as produced by cicflowmeter
            (keys are cicflowmeter column names).

        Returns
        -------
        dict with keys: label, score, is_attack, is_warning, all_scores
        """
        import pandas as pd

        # 1. Build a 1-row DataFrame with ALL scaler-known features in the
        #    correct order. The scaler was fitted on all 72 features BEFORE
        #    feature selection, so we must replicate that step.
        scaler_features: list[str] = list(self.scaler.feature_names_in_)
        row = {feat: self._safe_float(flow_data.get(feat, 0.0))
               for feat in scaler_features}
        X_full = pd.DataFrame([row], columns=scaler_features)

        # 2. Scale the full 72-feature vector
        X_scaled_full = self.scaler.transform(X_full)  # (1, 72)

        # 3. Select the 20 model-specific features *from the scaled array*
        feat_indices = [scaler_features.index(
            f) for f in self.selected_features]
        X_scaled = X_scaled_full[:, feat_indices].astype(np.float32)  # (1, 20)

        # 4. Append to sequence buffer
        self.buffer.append(X_scaled[0])
        
        if len(self.buffer) < self.seq_len:
            return {
                "label": "BUFFERING",
                "score": 0.0,
                "is_attack": False,
                "is_warning": False,
                "all_scores": {cls: 0.0 for cls in self.class_names},
                "status": f"{len(self.buffer)}/{self.seq_len}"
            }

        # 5. Inference
        with torch.no_grad():
            # Stack buffer into shape (1, seq_len, n_features)
            X_seq = np.stack(self.buffer)[np.newaxis, ...]
            tensor = torch.tensor(
                X_seq, dtype=torch.float32).to(self.device)
            # (1, n_classes)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (n_classes,)

        # 5. Decode
        pred_idx = int(np.argmax(probs))
        label = self.class_names[pred_idx]
        score = float(probs[pred_idx])
        all_scores = {cls: float(p) for cls, p in zip(self.class_names, probs)}

        is_attack = label.upper() != "BENIGN"
        # Warning: model is unsure (attack class probability between thresholds)
        is_warning = is_attack and WARNING_LOW <= score < ATTACK_THRESH

        return {
            "label": label,
            "score": score,
            "is_attack": is_attack,
            "is_warning": is_warning,
            "all_scores": all_scores,
            "status": "ready"
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value) -> float:
        """Convert value to float, replacing inf/NaN with 0."""
        try:
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return v
        except (TypeError, ValueError):
            return 0.0
