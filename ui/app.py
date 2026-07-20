"""
ui.app — OpenMedical Gradio Web Interface
==========================================
Industrial-grade chest X-ray AI analysis platform UI.
Orchestrates the CT model, tools, and DeepSeek agent.

Usage:
    python -m ui.app
    # or from main.py
"""

from io import BytesIO
import numpy as np
from PIL import Image
import gradio as gr

from ct_model import get_inference, ChestXrayInference
from tools import get_ct_analysis_tool
from agentic import get_agent


# ============================================================
# Custom CSS
# ============================================================
CUSTOM_CSS = """
:root {
    --primary: #1a73e8;
    --primary-dark: #1557b0;
    --success: #0d904f;
    --warning: #e67e22;
    --danger: #c0392b;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --border: #dee2e6;
    --text: #212529;
    --text-secondary: #6c757d;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.header-container {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 24px rgba(13, 71, 161, 0.2);
}
.header-title {
    color: white;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header-title span {
    opacity: 0.85;
    font-weight: 400;
    font-size: 16px;
    margin-left: 10px;
}

.health-index-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 16px 28px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}
.health-index-value {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}
.health-index-label {
    font-size: 13px;
    color: rgba(255,255,255,0.8);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}
.health-index-bar {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    margin-top: 8px;
    background: rgba(255,255,255,0.2);
    overflow: hidden;
}
.health-index-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease, background 0.6s ease;
}

.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.status-loaded { background: #e8f5e9; color: #2e7d32; }
.status-not-loaded { background: #fff3e0; color: #e65100; }

/* ---- Code-block results (monospace, pre-wrap) ---- */
.results-container {
    background: var(--card-bg);
    color: #212529;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border);
    font-family: 'Cascadia Code', 'Consolas', 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    max-height: 600px;
    overflow-y: auto;
}

/* ---- Agent markdown output (rich text, readable dark-on-light) ---- */
.agent-container {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid var(--border);
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.7;
    max-height: 700px;
    overflow-y: auto;
}
.agent-container,
.agent-container p,
.agent-container li,
.agent-container td,
.agent-container th,
.agent-container span,
.agent-container div {
    color: #212529 !important;
}
.agent-container h1, .agent-container h2, .agent-container h3,
.agent-container h4, .agent-container h5, .agent-container h6 {
    color: #1a237e !important;
}
.agent-container strong, .agent-container b {
    color: #0d47a1 !important;
}
.agent-container em, .agent-container i {
    color: #37474f !important;
}
.agent-container blockquote {
    border-left: 4px solid #1a73e8;
    background: #e8f0fe;
    padding: 10px 16px;
    margin: 12px 0;
}
.agent-container blockquote,
.agent-container blockquote p {
    color: #37474f !important;
}
.agent-container code {
    color: #c62828 !important;
    background: #ffebee;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}
.agent-container pre {
    background: #263238;
    color: #eceff1 !important;
    padding: 14px;
    border-radius: 8px;
    overflow-x: auto;
}
.agent-container pre code {
    color: #eceff1 !important;
    background: transparent;
}
.agent-container ul, .agent-container ol {
    padding-left: 24px;
}
.agent-container hr {
    border-color: var(--border);
}

.primary-btn {
    background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(26, 115, 232, 0.4) !important;
}

.tab-nav button {
    border-radius: 10px 10px 0 0 !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

.footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 12px;
    margin-top: 24px;
    padding: 16px;
    border-top: 1px solid var(--border);
}
"""


# ============================================================
# UI Callback Functions
# ============================================================
def on_load_model(progress=gr.Progress()):
    """Load the CT model via the tool + inference engine."""
    progress(0, desc="Loading CT model...")
    try:
        tool = get_ct_analysis_tool()
        info = tool.initialize()
        progress(1.0, desc="Model loaded!")

        status_html = (
            '<span class="status-badge status-loaded">✅ Model Loaded</span>'
            f'<br><small>AUC: {info["best_auc"]:.4f} | Device: {info["device"]} | '
            f'T={info["temperature"]} | {info["image_size"]}px | '
            f'14 pathologies</small>'
        )
        return (
            status_html,
            gr.update(interactive=True),    # run_btn
            gr.update(interactive=True),    # agent_btn
        )
    except Exception as e:
        status_html = (
            '<span class="status-badge status-not-loaded">❌ Failed</span>'
            f'<br><small>{str(e)}</small>'
        )
        return (
            status_html,
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def on_run_analysis(image):
    """Run CT analysis on the uploaded image."""
    if image is None:
        return (
            "⚠️ Please upload a chest X-ray image first.",
            "## --\n\n*Upload an image and click Run Analysis*",
            "",
            None,
        )

    try:
        # Normalize image to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        tool = get_ct_analysis_tool()
        if not tool.is_ready:
            return (
                "⚠️ Please load the model first using the 'Load Model' button.",
                "## --\n\n*Model not loaded*",
                "",
                None,
            )

        results = tool.run_from_pil(pil_image)
        engine = get_inference()
        report = engine.format_report(results)

        # Health Index display
        hi = results["health_index"]
        if hi >= 80:
            bar_color, emoji = "#27ae60", "🟢"
        elif hi >= 50:
            bar_color, emoji = "#e67e22", "🟡"
        else:
            bar_color, emoji = "#c0392b", "🔴"

        health_html = (
            f"## {emoji} Health Index: {hi}/100\n\n"
            f'<div style="width:100%;height:12px;border-radius:6px;'
            f'background:#e9ecef;overflow:hidden;margin:8px 0 16px 0;">'
            f'<div style="width:{hi}%;height:100%;border-radius:6px;'
            f'background:{bar_color};transition:width 0.8s ease;"></div>'
            f'</div>\n\n'
            f'**Risk Level:** <span style="color:{bar_color};font-weight:700;">'
            f'{results["risk_level"]}</span> | '
            f'**Findings:** {results["num_positive"]} detected | '
            f'**Borderline:** {results["num_borderline"]}'
        )

        risk_display = (
            f'<span style="color:{bar_color};font-weight:700;font-size:18px;">'
            f'{results["risk_level"]} RISK</span> — '
            f'{results["num_positive"]} finding(s) detected'
        )

        return (
            f"```\n{report}\n```",
            health_html,
            risk_display,
            results,
        )
    except Exception as e:
        import traceback
        return (
            f"❌ **Error:** {e}\n\n```\n{traceback.format_exc()}\n```",
            "## --\n\n*Error during analysis*",
            "ERROR",
            None,
        )


def on_agent_analysis(current_results):
    """Run DeepSeek agent analysis on the CT results."""
    if current_results is None:
        return "⚠️ Please run an analysis first before requesting agent interpretation."

    agent = get_agent()
    if not agent.is_configured:
        return (
            "❌ **Configuration Error:** DEEPSEEK_API_KEY not found.\n\n"
            "Please add your DeepSeek API key to `agentic/.env`:\n"
            "```\nDEEPSEEK_API_KEY=sk-your-key-here\n```"
        )

    response = agent.clinical_interpretation(current_results, include_disclaimer=True)
    return response


# ============================================================
# Build UI
# ============================================================
def build_app() -> gr.Blocks:
    """Build the Gradio Blocks application."""

    # OpenMP workaround for Windows + PyTorch
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    with gr.Blocks(title="OpenMedical — Chest X-ray AI Analysis") as app:

        # ---- State ----
        current_results_state = gr.State(None)

        # ---- HEADER ----
        with gr.Row(elem_classes=["header-container"]):
            with gr.Column(scale=3):
                gr.HTML(
                    '<div class="header-title">🏥 OpenMedical'
                    '<span>| Chest X-ray AI Analysis Platform</span></div>'
                    '<div style="color:rgba(255,255,255,0.7);font-size:13px;margin-top:6px;">'
                    'ConvNeXt-Tiny + ASL + cRT | NIH ChestX-ray14 | AUC 0.806 | 512px'
                    '</div>'
                )
            with gr.Column(scale=1):
                gr.HTML(
                    '<div class="health-index-card">'
                    '<div class="health-index-value">--</div>'
                    '<div class="health-index-label">Health Index</div>'
                    '<div class="health-index-bar">'
                    '<div class="health-index-bar-fill" style="width:0%;background:#95a5a6;"></div>'
                    '</div></div>'
                )

        # ---- STATUS ROW ----
        with gr.Row():
            with gr.Column(scale=1):
                risk_level_display = gr.HTML("")
            with gr.Column(scale=1):
                model_status = gr.HTML(
                    '<span class="status-badge status-not-loaded">⏳ Model Not Loaded</span>'
                )

        # ---- TABS ----
        with gr.Tabs(elem_classes=["tabs"]):
            # ============ CHEST CT TAB ============
            with gr.TabItem("🫁 Chest CT", id="chest_ct"):
                gr.Markdown(
                    "### Chest X-ray Pathology Detection\n"
                    "Upload a frontal chest X-ray image for AI-assisted analysis "
                    "covering **14 common thoracic pathologies**."
                )

                with gr.Row():
                    # LEFT: Controls
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ Model & Input")

                        load_btn = gr.Button(
                            "📦 Load Model",
                            variant="primary",
                            elem_classes=["primary-btn"],
                            size="lg",
                        )

                        image_input = gr.Image(
                            label="Upload Chest X-ray",
                            type="pil",
                            image_mode="RGB",
                            height=380,
                        )

                        with gr.Row():
                            run_btn = gr.Button(
                                "🔍 Run Analysis",
                                variant="primary",
                                elem_classes=["primary-btn"],
                                size="lg",
                                interactive=False,
                            )
                            agent_btn = gr.Button(
                                "🤖 Agent Analysis",
                                variant="secondary",
                                size="lg",
                                interactive=False,
                            )

                        gr.Markdown(
                            "<small>💡 **Tip:** Use frontal (PA/AP) chest X-rays for "
                            "best results. Trained on NIH ChestX-ray14 at 512×512px."
                            "</small>"
                        )

                    # RIGHT: Results
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📊 Analysis Results")

                        health_index_display = gr.Markdown(
                            "## --\n\n*Upload an image and click Run Analysis*"
                        )

                        results_display = gr.Markdown(
                            "> *Upload an image and click **Run Analysis** to see results.*",
                            elem_classes=["results-container"],
                        )

                # ---- Agent Analysis Section ----
                gr.Markdown("---")
                gr.Markdown("### 🤖 Agent Analysis (DeepSeek AI)")
                agent_output = gr.Markdown(
                    "> *Click **Agent Analysis** after running inference to get "
                    "AI-powered clinical interpretation.*",
                    elem_classes=["agent-container"],
                )

        # ---- FOOTER ----
        gr.HTML(
            '<div class="footer">'
            'OpenMedical v5 — ConvNeXt-Tiny + ASL + cRT | NIH ChestX-ray14 | '
            'AUC 0.806 | 512px | For research purposes only — NOT a medical device | '
            '© 2026 OpenMedical Research'
            '</div>'
        )

        # ---- EVENT BINDINGS ----
        load_btn.click(
            fn=on_load_model,
            outputs=[model_status, run_btn, agent_btn],
        )

        run_btn.click(
            fn=on_run_analysis,
            inputs=[image_input],
            outputs=[
                results_display,
                health_index_display,
                risk_level_display,
                current_results_state,
            ],
        )

        agent_btn.click(
            fn=on_agent_analysis,
            inputs=[current_results_state],
            outputs=[agent_output],
        )

    return app


# ============================================================
# Launch
# ============================================================
def launch():
    """Launch the Gradio application."""
    print("=" * 60)
    print("🏥 OpenMedical — Chest X-ray AI Analysis Platform")
    print("=" * 60)
    print("  Model:  ConvNeXt-Tiny + ASL + cRT (v5)")
    print("  Res:    512×512px | 14 pathologies")
    print("  Agent:  DeepSeek v4 Pro")
    print("=" * 60)

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    )


if __name__ == "__main__":
    launch()
