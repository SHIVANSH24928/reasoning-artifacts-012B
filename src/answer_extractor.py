"""
Answer Extraction Pipeline
==========================
Paper Section 3.4: "Answer Extraction Pipeline"

Implements the prioritized 3-tier extraction hierarchy:
- Tier 1 (regex_hash): Matches canonical GSM8K delimiter (####) with bracket tolerance.
- Tier 2 (regex_natural): Catches natural-language conclusion phrases.
- Tier 3 (Fallback): Extracts the terminal integer token in the generation trace.
"""

import re
from typing import Tuple, Optional

TIER1_PATTERN = re.compile(r'#{4}\s*\[?([+-]?\d+)\]?')
TIER2_PATTERN = re.compile(
    r'(?:final answer is|therefore\s+(?:the\s+)?(?:answer|total|result)\s+is|the answer is)\s*\$?([+-]?\d+)',
    flags=re.IGNORECASE
)
TIER3_PATTERN = re.compile(r'\b(\d+)\b')


def extract_answer_with_tier(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts the final numerical prediction and returns (answer, tier_level).
    tier_level is 1, 2, or 3, or None if extraction fails.
    """
    if not isinstance(text, str) or not text.strip():
        return None, None

    # Tier 1: Canonical hash delimiter with optional brackets (e.g. '#### 15' or '#### [15]')
    m = TIER1_PATTERN.search(text)
    if m:
        try:
            return int(m.group(1)), 1
        except ValueError:
            pass

    # Tier 2: Natural language phrases
    m = TIER2_PATTERN.search(text)
    if m:
        try:
            return int(m.group(1)), 2
        except ValueError:
            pass

    # Tier 3: Fallback integer token at end of generation trace
    tokens = TIER3_PATTERN.findall(text)
    if tokens:
        try:
            return int(tokens[-1]), 3
        except ValueError:
            pass

    return None, None


def extract_answer(text: str) -> Optional[int]:
    """Convenience helper returning only the extracted integer answer."""
    ans, _ = extract_answer_with_tier(text)
    return ans
