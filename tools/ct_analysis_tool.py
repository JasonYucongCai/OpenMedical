"""
tools.ct_analysis_tool — Chest X-ray CT Analysis as a callable agent tool
==========================================================================
Wraps the ConvNeXt-Tiny inference engine as a tool that can be invoked
by the DeepSeek agent or any other orchestrator.

Tool Interface:
    - name: "ct_analysis"
    - description: Human-readable description for the agent
    - run(image_path: str) -> dict: Execute analysis on an image file
    - run_from_pil(image: PIL.Image) -> dict: Execute on a PIL Image
    - to_prompt(results: dict) -> str: Convert results to agent-readable prompt
"""

from typing import Optional
from PIL import Image

from ct_model import get_inference, PATHOLOGY_LABELS


class CTAnalysisTool:
    """
    Chest X-ray CT Analysis Tool.

    Provides AI-powered chest X-ray pathology detection covering 14 common
    thoracic pathologies. Uses ConvNeXt-Tiny + ASL + cRT (v5 Industrial model).

    Exposed to the agent as a callable tool with a defined interface.
    """

    name = "ct_analysis"
    description = (
        "Analyze a chest X-ray image for 14 common thoracic pathologies "
        "(Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, "
        "Pneumonia, Pneumothorax). Returns per-pathology probabilities, "
        "a Health Index score (0-100), and clinical risk assessment."
    )

    def __init__(self):
        # Use singleton engine directly — model may already be loaded
        pass

    @property
    def is_ready(self) -> bool:
        engine = get_inference()
        return engine.is_loaded

    def initialize(self, checkpoint_path: Optional[str] = None) -> dict:
        """Load the underlying model. Must be called before run()."""
        engine = get_inference()
        if not engine.is_loaded:
            return engine.load_model(checkpoint_path)
        return engine.checkpoint_info

    def run(self, image_path: str) -> dict:
        if not self.is_ready:
            raise RuntimeError(
                "CT Analysis Tool not initialized. Call initialize() first."
            )
        return get_inference().predict_file(image_path)

    def run_from_pil(self, image: Image.Image) -> dict:
        if not self.is_ready:
            raise RuntimeError(
                "CT Analysis Tool not initialized. Call initialize() first."
            )
        return get_inference().predict(image)

    @staticmethod
    def to_prompt(results: dict) -> str:
        """
        Convert CT analysis results into a structured prompt for the agent.

        This builds a detailed text block the DeepSeek agent can use for
        clinical interpretation.
        """
        classes_str = ""
        for c in results["classes"]:
            classes_str += (
                f"  - {c['label']}: prob={c['probability']:.4f}, "
                f"threshold={c['threshold']:.3f}, "
                f"prediction={c['prediction']}, confidence={c['confidence']}\n"
            )

        positive = results.get("positive_findings", [])
        pos_str = "\n".join(
            f"  - {c['label']} (prob={c['probability']:.3f}, "
            f"confidence={c['confidence']})"
            for c in positive
        ) if positive else "  None"

        borderline = results.get("borderline_findings", [])
        bdl_str = "\n".join(
            f"  - {c['label']} (prob={c['probability']:.3f})"
            for c in borderline
        ) if borderline else "  None"

        risks = results.get("risk_details", [])
        risk_str = "\n".join(f"  • {d}" for d in risks) if risks else "  No critical risks"

        return f"""You are an expert radiologist AI assistant. Analyze the following chest X-ray model predictions and provide a clinical interpretation.

**Patient Chest X-ray Analysis Results:**
- Health Index: {results['health_index']}/100 (higher = healthier)
- Number of positive findings: {results['num_positive']}
- Number of borderline cases: {results['num_borderline']}
- Risk Level: {results['risk_level']}

**Per-Pathology Predictions:**
{classes_str}
**Detected Findings:**
{pos_str}

**Borderline Findings (below threshold but elevated):**
{bdl_str}

**Clinical Risk Notes:**
{risk_str}

Please provide:
1. **Summary**: A concise summary of the key findings (2-3 sentences)
2. **Differential Diagnosis**: What conditions should be considered given these findings?
3. **Recommendations**: What follow-up actions would you suggest?
4. **Limitations**: Note any limitations of this AI analysis (model AUC ~0.80, single-view analysis, etc.)
5. **Confidence Assessment**: How reliable is this prediction overall?

IMPORTANT: This is an AI research tool, NOT a medical device. Always emphasize that findings should be verified by a board-certified radiologist."""
