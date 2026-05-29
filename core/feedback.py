"""
Phase 3: LLM Feedback Generation

Takes a FaultReport and generates:
  - Plain-language explanation of each fault
  - Root cause in simple terms (no jargon)
  - Specific drills to fix each fault
  - A priority order (fix this first, then this)
  - A one-paragraph overall swing summary

Uses the Anthropic API (claude-sonnet-4-20250514).
Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

from .fault_engine import FaultReport


@dataclass
class DrillRecommendation:
    name: str
    description: str
    reps_or_duration: str
    focus_cue: str   # the one thing to feel/think during the drill


@dataclass
class FaultFeedback:
    fault_name: str
    display_name: str
    severity_label: str
    plain_explanation: str    # simple language, no biomechanics jargon
    what_you_feel: str        # what the golfer likely feels when this happens
    ball_flight_clue: str     # how they can self-diagnose from their shots
    drills: List[DrillRecommendation]
    priority: int             # 1 = fix first


@dataclass
class SwingFeedback:
    overall_summary: str
    priority_focus: str       # one sentence: "your most important fix is..."
    fault_feedback: List[FaultFeedback]
    raw_response: str = ""    # full LLM response for debugging

    def print_report(self):
        print("\n" + "=" * 60)
        print("SWING COACHING REPORT")
        print("=" * 60)
        print(f"\n{self.overall_summary}")
        print(f"\n🎯 PRIORITY FOCUS: {self.priority_focus}")

        for fb in self.fault_feedback:
            print(f"\n{'─' * 60}")
            print(f"#{fb.priority} [{fb.severity_label.upper()}] {fb.display_name}")
            print(f"\nWhat's happening:")
            print(f"  {fb.plain_explanation}")
            print(f"\nWhat you probably feel:")
            print(f"  {fb.what_you_feel}")
            print(f"\nHow to self-diagnose from your shots:")
            print(f"  {fb.ball_flight_clue}")
            print(f"\nDrills to fix it:")
            for i, drill in enumerate(fb.drills, 1):
                print(f"  {i}. {drill.name} ({drill.reps_or_duration})")
                print(f"     {drill.description}")
                print(f"     Feel: {drill.focus_cue}")


class FeedbackGenerator:
    """
    Generates coaching feedback from fault detection results using Claude.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-5-20250929"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def generate(
        self,
        report: FaultReport,
        club: str = "7i",
        max_faults: int = 3,
    ) -> SwingFeedback:
        """
        Generate coaching feedback for the top faults in the report.

        Parameters
        ----------
        report     : FaultReport from fault detection
        club       : club used (for context)
        max_faults : max number of faults to give feedback on (default 3)
                     Focus beats overwhelm — don't give 8 things to fix.
        """
        if not report.faults:
            return SwingFeedback(
                overall_summary="No significant faults detected compared to elite benchmarks. Focus on consistency and tempo.",
                priority_focus="Maintain your current mechanics and focus on repeating your best swings.",
                fault_feedback=[],
            )

        # Take top N faults by severity
        top_faults = report.faults[:max_faults]

        # Build the prompt
        prompt = self._build_prompt(top_faults, report, club)

        # Call the API
        raw = self._call_api(prompt)

        # Parse the response
        return self._parse_response(raw, top_faults)

    def _build_prompt(self, faults, report, club) -> str:
        fault_list = []
        for i, f in enumerate(faults, 1):
            fault_list.append(f"""
FAULT {i}: {f.display_name}
  Severity: {f.severity_label} (score: {f.severity:.2f}/1.0)
  Phase: {f.phase}
  Measured value: {f.measured_value}
  Elite benchmark: {f.elite_benchmark}
  Biomechanical description: {f.description}
  Root cause: {f.root_cause}
  Ball flight effect: {f.ball_flight}
  Research source: {f.source}
""")

        return f"""You are an expert PGA-certified golf instructor analyzing a swing from biomechanical data.

The golfer was hitting a {club}. The following faults were detected by comparing their swing against PGA Tour benchmarks.

DETECTED FAULTS (ordered by severity, most severe first):
{''.join(fault_list)}

Your task: Generate practical coaching feedback in JSON format with this exact structure:

{{
  "overall_summary": "2-3 sentence plain-language summary of the swing. Be honest but encouraging. Mention what's good if severity scores are low.",
  "priority_focus": "One sentence: the single most important thing to work on first.",
  "faults": [
    {{
      "fault_name": "exact fault name from input",
      "plain_explanation": "Explain what's happening in simple language a 15-year-old golfer would understand. No biomechanics jargon. 2-3 sentences.",
      "what_you_feel": "What does the golfer likely feel when making this mistake? What sensation are they experiencing that feels 'normal' to them?",
      "ball_flight_clue": "How can they identify this fault from their ball flight alone, without a camera?",
      "drills": [
        {{
          "name": "Drill name",
          "description": "Clear step-by-step instructions. Be specific.",
          "reps_or_duration": "e.g. '10 reps', '5 minutes', '20 balls'",
          "focus_cue": "The ONE feeling or thought to focus on during the drill"
        }},
        {{
          "name": "Second drill name",
          "description": "Clear step-by-step instructions.",
          "reps_or_duration": "e.g. '10 reps'",
          "focus_cue": "The ONE feeling or thought"
        }}
      ]
    }}
  ]
}}

Rules:
- Give exactly {len(faults)} fault objects, in the same order as the input
- Give exactly 2 drills per fault
- Use simple, conversational language — not textbook language
- Drills must be practical range drills, not gym exercises
- Focus cues must be simple feels, not technical positions
- Be specific with drill instructions (what to do with feet, where to feel it, etc.)
- Output ONLY the JSON. No preamble, no explanation, no markdown fences."""

    def _call_api(self, prompt: str) -> str:
        if not self.api_key:
            return self._offline_fallback()

        payload = json.dumps({
            "model": self.MODEL,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            self.API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"API error {e.code}: {body}")

    def _parse_response(self, raw: str, faults) -> SwingFeedback:
        try:
            # Strip any accidental markdown fences
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)

            fault_feedbacks = []
            for i, (fault_data, original_fault) in enumerate(
                zip(data.get("faults", []), faults), 1
            ):
                drills = []
                for d in fault_data.get("drills", []):
                    drills.append(DrillRecommendation(
                        name=d.get("name", ""),
                        description=d.get("description", ""),
                        reps_or_duration=d.get("reps_or_duration", ""),
                        focus_cue=d.get("focus_cue", ""),
                    ))

                fault_feedbacks.append(FaultFeedback(
                    fault_name=fault_data.get("fault_name", original_fault.name),
                    display_name=original_fault.display_name,
                    severity_label=original_fault.severity_label,
                    plain_explanation=fault_data.get("plain_explanation", ""),
                    what_you_feel=fault_data.get("what_you_feel", ""),
                    ball_flight_clue=fault_data.get("ball_flight_clue", ""),
                    drills=drills,
                    priority=i,
                ))

            return SwingFeedback(
                overall_summary=data.get("overall_summary", ""),
                priority_focus=data.get("priority_focus", ""),
                fault_feedback=fault_feedbacks,
                raw_response=raw,
            )

        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, return a basic fallback with the raw text
            return SwingFeedback(
                overall_summary="Feedback generated — see raw response below.",
                priority_focus=f"Focus on: {faults[0].display_name if faults else 'consistency'}",
                fault_feedback=[],
                raw_response=raw,
            )

    def _offline_fallback(self) -> str:
        """Returns a placeholder when no API key is set."""
        return json.dumps({
            "overall_summary": "API key not set. Set ANTHROPIC_API_KEY environment variable to get AI feedback.",
            "priority_focus": "Set your API key to get personalized feedback.",
            "faults": []
        })