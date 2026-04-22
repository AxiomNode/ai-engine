"""Lightweight, rule-based jailbreak / prompt-injection classifier.

This module implements the "lightweight jailbreak classifier" backlog item
documented in
``docs/operations/threat-model.md`` (STRIDE row "Tampering" for ``ai-engine``)
and complements the rules in ``docs/guides/ai-prompting-rules.md``.

Goals
-----
* Run before any LLM call and reject obviously malicious prompts with a
  deterministic, dependency-free heuristic (no remote model required).
* Stay observable: every block is recorded in :mod:`ai_engine.observability`
  with a machine-readable ``safety_block_reason``.
* Stay tunable: thresholds and toggles can be overridden through environment
  variables without code changes.

The classifier is intentionally conservative — false positives are preferred
to false negatives for the public surface area of the engine. Callers can
opt out by setting ``AI_ENGINE_SAFETY_ENABLED=false``.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class _Pattern:
    """Single jailbreak heuristic with a category and severity weight."""

    name: str
    category: str
    weight: float
    regex: re.Pattern[str]


@dataclass(frozen=True)
class SafetyDecision:
    """Outcome of evaluating a prompt against the jailbreak classifier."""

    verdict: str  # "allow" | "flag" | "block"
    score: float
    reasons: tuple[str, ...] = ()
    matched_patterns: tuple[str, ...] = ()
    matched_categories: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"

    @property
    def primary_reason(self) -> str:
        return self.reasons[0] if self.reasons else "ok"


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


# Patterns are grouped by intent. Weights are calibrated so a single
# strong signal (>=1.0) triggers a block, while several weaker signals can
# combine to escalate from "flag" to "block".
_PATTERNS: tuple[_Pattern, ...] = (
    # ---- Instruction override / role takeover -----------------------
    _Pattern(
        name="ignore_previous_instructions",
        category="instruction_override",
        weight=1.0,
        regex=_compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^\n]{0,40}"
            r"\b(previous|prior|above|earlier|all)\b[^\n]{0,40}"
            r"\b(instructions?|prompts?|rules?|messages?|directives?)\b"
        ),
    ),
    _Pattern(
        name="ignore_previous_instructions_es",
        category="instruction_override",
        weight=1.0,
        regex=_compile(
            r"\b(ignora|olvida|omite|salta|sobrescribe|anula)\b[^\n]{0,40}"
            r"\b(las|tus|todas|todo|los)?\s*"
            r"(instrucciones|reglas|directrices|mensajes|prompts?)\b"
        ),
    ),
    _Pattern(
        name="role_takeover",
        category="instruction_override",
        weight=0.9,
        regex=_compile(
            r"\b(you are now|act as|pretend to be|from now on you are|"
            r"eres ahora|actua como|finge ser|a partir de ahora eres)\b[^\n]{0,80}"
            r"\b(dan|jailbreak|admin|root|system|developer mode|modo desarrollador)\b"
        ),
    ),
    _Pattern(
        name="developer_mode",
        category="instruction_override",
        weight=0.9,
        regex=_compile(
            r"\b(developer\s*mode|dev\s*mode|modo\s*desarrollador|"
            r"jailbreak\s*mode|do\s*anything\s*now|dan\s*mode)\b"
        ),
    ),
    # ---- System prompt exfiltration ---------------------------------
    _Pattern(
        name="reveal_system_prompt",
        category="system_prompt_leak",
        weight=1.0,
        regex=_compile(
            r"\b(reveal|show|print|repeat|expose|leak|dump)\b[^\n]{0,40}"
            r"\b(system\s*prompt|hidden\s*prompt|initial\s*prompt|"
            r"original\s*instructions|instructions\s*above|secret\s*prompt|"
            r"prompt\s*del\s*sistema|instrucciones\s*originales)\b"
        ),
    ),
    _Pattern(
        name="reveal_secrets",
        category="system_prompt_leak",
        weight=1.0,
        regex=_compile(
            r"\b(reveal|show|print|leak|expose|dump|give\s+me|dame|muestra|revela)\b"
            r"[^\n]{0,40}"
            r"\b(api[_\s-]?key|secret|password|contrase[nñ]a|token|credentials?|"
            r"credenciales?|env\s*vars?|environment\s*variables?)\b"
        ),
    ),
    # ---- Delimiter / context escape ---------------------------------
    _Pattern(
        name="closing_system_tag",
        category="encoding_evasion",
        weight=0.7,
        regex=_compile(
            r"(</\s*(system|s|assistant|sys|instructions?)\s*>|"
            r"<\|im_end\|>|<\|endoftext\|>|\[/INST\]|<<\s*/SYS\s*>>)"
        ),
    ),
    _Pattern(
        name="opening_system_tag",
        category="encoding_evasion",
        weight=0.5,
        regex=_compile(
            r"(<\s*(system|sys|instructions?|assistant)\s*>|"
            r"<\|im_start\|>|\[INST\]|<<\s*SYS\s*>>)"
        ),
    ),
    _Pattern(
        name="role_label_injection",
        category="encoding_evasion",
        weight=0.5,
        regex=_compile(
            r"^\s*(system|assistant|user|developer|sistema)\s*[:>][\s\S]{0,8}"
        ),
    ),
    # ---- Dangerous content / policy bypass --------------------------
    _Pattern(
        name="weapons_or_illicit",
        category="dangerous_request",
        weight=1.0,
        regex=_compile(
            r"\b(how\s*to\s*(make|build|create)|instructions?\s*for|"
            r"como\s*(hacer|fabricar|construir))\b[^\n]{0,60}"
            r"\b(bomb|explosive|malware|ransomware|virus|exploit|"
            r"weapon|firearm|silencer|bomba|explosivo|arma)\b"
        ),
    ),
    _Pattern(
        name="bypass_safety",
        category="instruction_override",
        weight=0.8,
        regex=_compile(
            r"\b(bypass|disable|turn\s*off|circumvent|evita|desactiva|"
            r"salta|elimina)\b[^\n]{0,40}"
            r"\b(safety|guardrails?|filters?|content\s*policy|moderation|"
            r"restricciones|filtro|seguridad|moderaci[oó]n)\b"
        ),
    ),
)


def _normalise(text: str) -> str:
    """Normalise text to defeat trivial obfuscation attempts."""
    if not text:
        return ""
    # Collapse zero-width characters and decompose accents so the regex
    # set written without diacritics still matches Spanish input.
    cleaned = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
        and unicodedata.category(ch) not in {"Cf", "Co"}
    )
    # Replace separator-style obfuscation ("i.g.n.o.r.e") with a single space.
    cleaned = re.sub(r"(?<=\w)[.\-_*~`]+(?=\w)", "", cleaned)
    return cleaned


@dataclass
class JailbreakClassifier:
    """Heuristic classifier that detects prompt-injection patterns."""

    block_threshold: float = 1.0
    flag_threshold: float = 0.4
    enabled: bool = True
    patterns: tuple[_Pattern, ...] = field(default=_PATTERNS)

    def classify(self, text: str) -> SafetyDecision:
        """Return a :class:`SafetyDecision` for *text*."""
        if not self.enabled or not text or not text.strip():
            return SafetyDecision(verdict="allow", score=0.0)

        normalised = _normalise(text)
        score = 0.0
        reasons: list[str] = []
        matched: list[str] = []
        categories: list[str] = []

        for pattern in self.patterns:
            if pattern.regex.search(normalised):
                score += pattern.weight
                matched.append(pattern.name)
                if pattern.category not in categories:
                    categories.append(pattern.category)
                reasons.append(f"{pattern.category}:{pattern.name}")

        # Cap to a known range so external metrics stay bounded.
        score = round(min(score, 5.0), 4)

        if score >= self.block_threshold:
            verdict = "block"
        elif score >= self.flag_threshold:
            verdict = "flag"
        else:
            verdict = "allow"

        return SafetyDecision(
            verdict=verdict,
            score=score,
            reasons=tuple(reasons),
            matched_patterns=tuple(matched),
            matched_categories=tuple(categories),
        )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_DEFAULT_CLASSIFIER: JailbreakClassifier | None = None


def get_default_classifier() -> JailbreakClassifier:
    """Return the process-wide classifier configured from environment."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = JailbreakClassifier(
            block_threshold=_env_float("AI_ENGINE_SAFETY_BLOCK_THRESHOLD", 1.0),
            flag_threshold=_env_float("AI_ENGINE_SAFETY_FLAG_THRESHOLD", 0.4),
            enabled=_env_bool("AI_ENGINE_SAFETY_ENABLED", True),
        )
    return _DEFAULT_CLASSIFIER


def reset_default_classifier() -> None:
    """Drop the cached classifier so the next call rereads env vars (tests)."""
    global _DEFAULT_CLASSIFIER
    _DEFAULT_CLASSIFIER = None


def evaluate_user_prompt(text: str) -> SafetyDecision:
    """Evaluate *text* with the default classifier."""
    return get_default_classifier().classify(text)


def iter_pattern_names(categories: Iterable[str] | None = None) -> list[str]:
    """Return the names of registered patterns (optionally filtered)."""
    allowed = set(categories) if categories else None
    return [p.name for p in _PATTERNS if allowed is None or p.category in allowed]
