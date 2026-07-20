"""
ct_model.inference — Model loading, preprocessing, inference, results formatting
================================================================================
Handles checkpoint loading, image transforms, temperature-scaled inference,
per-class thresholding, health index computation, and clinical risk assessment.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .model import ConvNeXtChestXray, PATHOLOGY_LABELS, NUM_CLASSES

# ---- Configuration (matching training) ----
IMAGE_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
BEST_TEMPERATURE = 0.7

# Per-class optimal thresholds (from v5 validation, T=0.7 temperature-scaled)
OPTIMAL_THRESHOLDS = np.array([
    0.20, 0.25, 0.15, 0.15, 0.30, 0.10, 0.10,
    0.95, 0.25, 0.15, 0.15, 0.15, 0.15, 0.20,
], dtype=np.float64)

# Clinical severity weights for Health Index (higher = more clinically significant)
SEVERITY_WEIGHTS = {
    "Pneumothorax": 1.0,  "Pneumonia": 0.9,  "Mass": 0.85,
    "Edema": 0.8,         "Consolidation": 0.75, "Effusion": 0.65,
    "Atelectasis": 0.55,  "Cardiomegaly": 0.6, "Emphysema": 0.5,
    "Fibrosis": 0.5,      "Infiltration": 0.45, "Nodule": 0.4,
    "Pleural_Thickening": 0.35, "Hernia": 0.3,
}

# ---- Image Transform ----
_inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================================================
# Checkpoint Path Resolution
# ==========================================================================
def _resolve_checkpoint_path(explicit_path: Optional[str] = None) -> str:
    """
    Resolve the checkpoint path for v5_stage2_latest.pth.
    Checks multiple locations in order:
        1. Explicit path (if provided)
        2. ct_model/checkpoints/  (preferred restructured location)
        3. Original training location (20260718 CT Prediction/checkpoints/)
    """
    if explicit_path and Path(explicit_path).exists():
        return explicit_path

    # Candidates in order of preference
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "ct_model" / "checkpoints" / "v5_stage2_latest.pth",
        project_root / "20260718 CT Prediction" / "checkpoints" / "v5_stage2_latest.pth",
        Path(r"c:\Deepin\Programming\20260718 CT Prediction\checkpoints\v5_stage2_latest.pth"),
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        f"Checkpoint 'v5_stage2_latest.pth' not found. Checked:\n" +
        "\n".join(f"  • {c}" for c in candidates)
    )


# ==========================================================================
# Inference Engine
# ==========================================================================
class ChestXrayInference:
    """
    End-to-end chest X-ray inference engine.

    Usage:
        engine = ChestXrayInference()
        engine.load_model()
        results = engine.predict(image)        # PIL Image
        results = engine.predict_file(path)     # File path
        report  = engine.format_report(results) # Text report
    """

    def __init__(self):
        self._model: Optional[ConvNeXtChestXray] = None
        self._device: Optional[torch.device] = None
        self._loaded = False
        self._checkpoint_info: dict = {}

    # ---- Properties ----
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return str(self._device) if self._device else "N/A"

    @property
    def checkpoint_info(self) -> dict:
        return dict(self._checkpoint_info)

    # ---- Device Detection ----
    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ---- Model Loading ----
    def load_model(self, checkpoint_path: Optional[str] = None) -> dict:
        """
        Load the ConvNeXt-Tiny model from checkpoint.

        Args:
            checkpoint_path: Optional explicit path to .pth file.

        Returns:
            dict with checkpoint metadata (auc, epoch, stage, device, etc.)
        """
        ckpt_path = _resolve_checkpoint_path(checkpoint_path)
        self._device = self._get_device()

        print(f"[CT Model] Device: {self._device}")
        print(f"[CT Model] Loading: {ckpt_path}")

        self._model = ConvNeXtChestXray(pretrained=False).to(self._device)
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()
        self._loaded = True

        self._checkpoint_info = {
            "path": ckpt_path,
            "best_auc": ckpt.get("best_auc", float("nan")),
            "epoch": ckpt.get("epoch", -1),
            "stage": ckpt.get("stage", -1),
            "device": str(self._device),
            "temperature": BEST_TEMPERATURE,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
        }
        print(f"[CT Model] Loaded — AUC: {self._checkpoint_info['best_auc']:.4f}")
        return self.checkpoint_info

    # ---- Inference ----
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single chest X-ray (PIL Image, RGB).

        Returns:
            dict with keys:
                health_index, temperature, classes (list of dicts),
                positive_findings, borderline_findings,
                num_positive, num_borderline,
                risk_level, risk_details, critical_missed, high_conf_positives
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        img_tensor = _inference_transform(image).unsqueeze(0).to(self._device)

        # Forward pass with temperature scaling
        with torch.no_grad():
            logits = self._model(img_tensor)
            probs = torch.sigmoid(logits / BEST_TEMPERATURE).cpu().numpy()[0]
            preds = (probs >= OPTIMAL_THRESHOLDS).astype(int)

        # Build per-class results
        classes = []
        for i, label in enumerate(PATHOLOGY_LABELS):
            p = float(probs[i])
            thresh = float(OPTIMAL_THRESHOLDS[i])
            pred = int(preds[i])

            if p >= 0.8:
                confidence = "HIGH"
            elif p >= 0.6:
                confidence = "MEDIUM"
            elif p >= thresh:
                confidence = "LOW"
            else:
                confidence = "NEGATIVE"

            classes.append({
                "label": label,
                "probability": round(p, 4),
                "threshold": thresh,
                "prediction": "Positive" if pred == 1 else "Negative",
                "pred_raw": pred,
                "confidence": confidence,
                "severity_weight": SEVERITY_WEIGHTS[label],
            })

        # ---- Health Index (0-100, higher = healthier) ----
        # Harmonic mean of per-class "healthiness" (1 - probability)
        # If even one pathology has high probability, the harmonic mean
        # drops sharply (since 1/(1-p) → ∞ as p → 1), penalizing the index.
        eps = 1e-6
        healthiness = np.maximum(1.0 - probs, eps)
        harmonic_mean = NUM_CLASSES / np.sum(1.0 / healthiness)
        health_index = round(100.0 * float(harmonic_mean), 1)

        # ---- Findings ----
        positive_findings = [c for c in classes if c["pred_raw"] == 1]
        borderline_findings = [
            c for c in classes
            if c["pred_raw"] == 0 and c["probability"] > 0.3
        ]

        # ---- Clinical Risk Assessment ----
        critical_labels = {"Pneumothorax", "Pneumonia", "Mass", "Edema"}
        critical_missed = [
            c for c in classes
            if c["pred_raw"] == 0 and c["probability"] > 0.3
            and c["label"] in critical_labels
        ]
        high_conf_positives = [
            c for c in classes
            if c["pred_raw"] == 1 and c["confidence"] == "HIGH"
            and c["label"] in critical_labels
        ]

        risk_level = "LOW"
        risk_details = []
        if critical_missed:
            risk_level = "HIGH"
            names = ", ".join(c["label"] for c in critical_missed)
            risk_details.append(f"CRITICAL: Possible missed finding(s): {names}")
        if high_conf_positives:
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
            hp_desc = ", ".join(
                f"{c['label']} ({c['probability']:.2f})"
                for c in high_conf_positives
            )
            risk_details.append(f"High-confidence detection(s): {hp_desc}")
        if len(positive_findings) >= 3:
            if risk_level == "LOW":
                risk_level = "MEDIUM"
            risk_details.append(f"Multiple findings detected ({len(positive_findings)})")

        return {
            "health_index": health_index,
            "temperature": BEST_TEMPERATURE,
            "classes": classes,
            "positive_findings": positive_findings,
            "borderline_findings": borderline_findings,
            "num_positive": len(positive_findings),
            "num_borderline": len(borderline_findings),
            "risk_level": risk_level,
            "risk_details": risk_details,
            "critical_missed": critical_missed,
            "high_conf_positives": high_conf_positives,
        }

    def predict_file(self, image_path: str) -> dict:
        """Run inference on an image file path."""
        img = Image.open(image_path).convert("RGB")
        return self.predict(img)

    # ---- Formatting ----
    @staticmethod
    def format_report(results: dict) -> str:
        """Convert inference results to a readable clinical report string."""
        lines = [
            "=" * 70,
            "🔬 OpenMedical — Chest X-ray Analysis Report",
            "=" * 70,
            f"  Health Index: {results['health_index']}/100",
            f"  Temperature:  T={results['temperature']}",
            f"  Positive Findings: {results['num_positive']}",
            f"  Borderline Cases: {results['num_borderline']}",
            f"  Risk Level: {results['risk_level']}",
            "",
            f"{'Pathology':<22s} {'Prob':>7s}  {'Thresh':>7s}  {'Pred':>9s}  {'Confidence':>10s}",
            "-" * 70,
        ]

        for c in results["classes"]:
            bar = "█" * int(c["probability"] * 20) + "░" * (20 - int(c["probability"] * 20))
            lines.append(
                f"{c['label']:<22s} {c['probability']:7.4f}  {c['threshold']:7.3f}  "
                f"{c['prediction']:>9s}  {bar} {c['confidence']}"
            )

        lines += [
            "", "-" * 70, "🏥 CLINICAL ASSESSMENT", "-" * 70,
        ]
        if results["risk_details"]:
            for detail in results["risk_details"]:
                lines.append(f"  • {detail}")
        else:
            lines.append("  ✅ No significant clinical risks detected.")

        hi = results["health_index"]
        if hi >= 85:
            lines.append("  ✅ Overall: Largely normal.")
        elif hi >= 60:
            lines.append("  ⚠️  Overall: Some findings — consider review.")
        else:
            lines.append("  🔴 Overall: Multiple findings — radiology review recommended.")

        lines.append("=" * 70)
        return "\n".join(lines)


# ==========================================================================
# Singleton convenience
# ==========================================================================
_engine: Optional[ChestXrayInference] = None


def get_inference() -> ChestXrayInference:
    """Get or create the singleton inference engine."""
    global _engine
    if _engine is None:
        _engine = ChestXrayInference()
    return _engine


def load_model(checkpoint_path: Optional[str] = None) -> dict:
    """Convenience: load model into the singleton engine."""
    return get_inference().load_model(checkpoint_path)
