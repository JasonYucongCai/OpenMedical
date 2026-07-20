"""
agentic.deepseek_agent — DeepSeek AI Agent for clinical interpretation
========================================================================
Handles:
  - DeepSeek API client initialization (OpenAI-compatible SDK)
  - Agent prompt construction with tool context
  - Tool-augmented analysis (agent can invoke CT analysis tool)
  - Structured clinical interpretation responses

Uses deepseek-v4-pro (latest as of 2026-07).
deepseek-chat & deepseek-reasoner are DEPRECATED (removed 2026-07-24).
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---- Load .env from agentic/ folder ----
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Fallback: try project root
    load_dotenv(Path(__file__).parent.parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


class DeepSeekAgent:
    """
    DeepSeek AI clinical analysis agent.

    Uses OpenAI-compatible API to call deepseek-v4-pro for expert
    radiological interpretation of chest X-ray model predictions.

    Usage:
        agent = DeepSeekAgent()
        analysis = agent.analyze(prompt="...")
        # Or with tool context:
        analysis = agent.analyze_with_tools(tool_context="...", user_prompt="...")
    """

    MODEL = "deepseek-v4-pro"
    BASE_URL = "https://api.deepseek.com"
    SYSTEM_PROMPT = (
        "You are an expert radiologist AI assistant working with OpenMedical, "
        "a chest X-ray analysis platform. You interpret model predictions and "
        "provide clinical context. Always emphasize that this is an AI research "
        "tool and not a medical device. Findings should be verified by a "
        "board-certified radiologist. Be concise but thorough. "
        "Structure your response with clear sections: Summary, Differential "
        "Diagnosis, Recommendations, Limitations, and Confidence Assessment."
    )

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or DEEPSEEK_API_KEY
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        """Lazy-init the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "`openai` package required. Install: pip install openai"
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.BASE_URL,
            )
        return self._client

    def analyze(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Send a prompt to DeepSeek and return the AI analysis.

        Args:
            prompt: The user/analysis prompt
            temperature: Model temperature (0.0-1.0)

        Returns:
            AI-generated clinical interpretation text.
        """
        if not self._api_key:
            return (
                "❌ **Configuration Error:** DEEPSEEK_API_KEY not found.\n\n"
                "Please ensure `.env` in the `agentic/` folder contains:\n"
                "```\nDEEPSEEK_API_KEY=sk-your-key-here\n```"
            )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
                stream=False,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "401" in error_msg:
                return (
                    "❌ **Authentication Error:** Invalid or expired DeepSeek API key.\n"
                    "Please update `DEEPSEEK_API_KEY` in `agentic/.env`."
                )
            if "rate" in error_msg.lower() or "429" in error_msg:
                return (
                    "⏳ **Rate Limit:** DeepSeek API rate limit reached. "
                    "Please wait a moment and try again."
                )
            return f"❌ **DeepSeek API Error:** {error_msg}\n\nPlease check your API key and network connection."

    def analyze_with_tool_context(
        self,
        tool_context: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """
        Analyze with awareness of available tools.

        Args:
            tool_context: Description of available tools (from ToolRegistry)
            user_prompt:  The specific analysis request
            temperature:  Model temperature

        Returns:
            AI-generated analysis with tool awareness.
        """
        full_prompt = (
            f"You have access to the following tools for chest X-ray analysis:\n\n"
            f"{tool_context}\n\n"
            f"---\n\n"
            f"{user_prompt}"
        )
        return self.analyze(full_prompt, temperature)

    def clinical_interpretation(
        self,
        analysis_results: dict,
        include_disclaimer: bool = True,
    ) -> str:
        """
        Generate a full clinical interpretation from CT analysis results.

        Args:
            analysis_results: Output dict from CTAnalysisTool.run() or
                              ChestXrayInference.predict()
            include_disclaimer: Prepend medical disclaimer header

        Returns:
            Markdown-formatted clinical interpretation report.
        """
        from tools.ct_analysis_tool import CTAnalysisTool

        prompt = CTAnalysisTool.to_prompt(analysis_results)
        response = self.analyze(prompt)

        if include_disclaimer:
            header = (
                "## 🤖 DeepSeek AI Agent — Clinical Interpretation\n\n"
                "> ⚠️ **Disclaimer:** This is an AI research tool, NOT a medical "
                "device. All findings must be verified by a board-certified "
                "radiologist.\n\n---\n\n"
            )
            return header + response

        return response


# ---- Singleton ----
_agent: Optional[DeepSeekAgent] = None


def get_agent() -> DeepSeekAgent:
    """Get or create the singleton DeepSeek agent."""
    global _agent
    if _agent is None:
        _agent = DeepSeekAgent()
    return _agent
