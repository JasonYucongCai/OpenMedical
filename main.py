"""
main.py — OpenMedical Application Entry Point
===============================================
Single entry point to launch the complete OpenMedical platform.

Usage:
    python main.py                # Launch with Gradio UI
    python main.py --test         # Run a quick inference test
    python main.py --agentic      # Run agentic analysis on test image
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def run_test():
    """Quick end-to-end test: load model + run inference on sample image."""
    print("=" * 60)
    print("🧪 OpenMedical — Quick Test")
    print("=" * 60)

    from ct_model import load_model
    from tools import get_ct_analysis_tool

    # Load model
    info = load_model()
    print(f"  ✅ Model loaded — AUC: {info['best_auc']:.4f}")

    # Run on sample
    sample = _project_root / "test_images" / "sample_xray.png"
    if not sample.exists():
        print(f"  ⚠️  Sample not found: {sample}")
        return

    tool = get_ct_analysis_tool()
    results = tool.run(str(sample))

    print(f"  📊 Health Index: {results['health_index']}/100")
    print(f"  🔬 Findings:     {results['num_positive']} positive")
    print(f"  ⚠️  Risk Level:   {results['risk_level']}")
    print()

    for c in results["classes"]:
        if c["pred_raw"] == 1 or c["probability"] > 0.3:
            print(f"  {c['label']:<22s} prob={c['probability']:.4f}  "
                  f"pred={c['prediction']}  conf={c['confidence']}")

    print("\n✅ Test complete.")


def run_agentic():
    """Run agentic analysis via DeepSeek on the test image."""
    print("=" * 60)
    print("🤖 OpenMedical — Agentic Analysis Test")
    print("=" * 60)

    from ct_model import load_model
    from tools import get_ct_analysis_tool
    from agentic import get_agent

    # Load model
    info = load_model()
    print(f"  ✅ Model loaded — AUC: {info['best_auc']:.4f}")

    # Run inference
    sample = _project_root / "test_images" / "sample_xray.png"
    tool = get_ct_analysis_tool()
    results = tool.run(str(sample))
    print(f"  📊 Health Index: {results['health_index']}/100")

    # Agent analysis
    agent = get_agent()
    if not agent.is_configured:
        print("  ❌ DeepSeek API not configured. Set DEEPSEEK_API_KEY in agentic/.env")
        return

    print("  🤖 Calling DeepSeek agent...")
    interpretation = agent.clinical_interpretation(results)
    print("\n" + interpretation)
    print("\n✅ Agentic test complete.")


def main():
    """Parse args and launch."""
    args = sys.argv[1:]

    if "--test" in args:
        run_test()
    elif "--agentic" in args:
        run_agentic()
    else:
        # Default: launch UI
        from ui import launch
        launch()


if __name__ == "__main__":
    main()
