"""
Prompt Construction Module
==========================
Paper Section 3.2, 5, and Figure 1

Constructs prompts for evaluation under:
1. Flawed Front-Position Prompt (Phase 2 Confound):
   Prepends the rule without updating the closing interrogative syntax.
2. Symmetric Prompt (Phase 3.5 Instrument Fix):
   Preserves the original problem text intact and appends a single unified terminal directive:
   "What is the final total after applying the rule?"
"""

from typing import Dict


def build_flawed_front_prompt(problem_text: str, rule: str) -> str:
    """
    Phase 2 Flawed Prompt (Front Condition).
    The rule is prepended, leaving the original problem question at the prompt end.
    This triggers attentional recency bias and answer echoing (Finding 2).
    """
    return f"[Rule: {rule}]\n{problem_text}"


def build_symmetric_prompt(problem_text: str, rule: str, position: str = "front") -> str:
    """
    Phase 3.5 Symmetric Prompt.
    Preserves the problem text and appends an explicit unified closing question.

    Args:
        problem_text: The complete GSM8K problem text (including its original question).
        rule: The structural or lexical perturbation rule.
        position: 'front' (rule before problem) or 'back' (rule after problem).
    """
    closing_directive = "What is the final total after applying the rule?"
    
    if position.lower() == "front":
        return f"[Rule: {rule}]\n{problem_text}\n{closing_directive}"
    elif position.lower() == "back":
        return f"{problem_text}\n[Rule: {rule}]\n{closing_directive}"
    else:
        raise ValueError(f"Unknown position: {position}. Expected 'front' or 'back'.")


def build_all_conditions(problem_text: str, structural_rule: str, lexical_rule: str) -> Dict[str, str]:
    """Generates all 5 conditions used in Phase 3.5 evaluation."""
    return {
        "baseline": problem_text,
        "structural_front": build_symmetric_prompt(problem_text, structural_rule, position="front"),
        "structural_back": build_symmetric_prompt(problem_text, structural_rule, position="back"),
        "lexical_front": build_symmetric_prompt(problem_text, lexical_rule, position="front"),
        "lexical_back": build_symmetric_prompt(problem_text, lexical_rule, position="back"),
    }
