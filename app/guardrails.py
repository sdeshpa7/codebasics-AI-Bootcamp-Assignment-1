"""
app/guardrails.py

Input and output guardrails for the FinSolve RAG pipeline.

Input guardrails (checked before query reaches the LLM):
  1. Rate limiting        – max 20 queries per session (in-memory)
  2. Prompt injection     – block attempts to override system instructions / bypass RBAC
  3. PII (absolute)       – always reject Aadhar, PAN, bank account numbers (any role)
  4. PII (HR-sensitive)   – salary, leave balance, attendance etc. blocked for non-HR users
  5. Off-topic detection  – reject queries unrelated to FinSolve business domains

Output guardrails (checked before response is returned to the user):
  1. Grounding check       – flag figures/dates in the response not traceable to retrieved chunks
  2. Cross-role leakage    – flag response terms from collections the user can't access
  3. Source citation        – warn if the response doesn't cite any source document or page
"""

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# Shared data model
# ===========================================================================

@dataclass
class GuardrailResult:
    """Result of a single input guardrail check."""
    passed: bool
    violation: str | None = None        # machine-readable tag
    message: str | None = None          # user-facing refusal message
    sanitized_query: str | None = None  # query with PII redacted (future use)


@dataclass
class OutputGuardrailResult:
    """Result of running all output guardrails against a response."""
    passed: bool                         # False if any hard block is triggered
    violations: list[str] = field(default_factory=list)   # hard blocks
    warnings: list[str]  = field(default_factory=list)    # soft warnings appended to response
    response: str = ""                   # (possibly annotated) final response


# ===========================================================================
# ── INPUT GUARDRAILS ────────────────────────────────────────────────────────
# ===========================================================================

# ── 1. Session rate limiter ─────────────────────────────────────────────────

_SESSION_LIMIT = 20
_session_counters: dict[str, int] = {}   # session_id → query count


def _check_rate_limit(session_id: str) -> GuardrailResult:
    count = _session_counters.get(session_id, 0) + 1
    _session_counters[session_id] = count
    if count > _SESSION_LIMIT:
        return GuardrailResult(
            passed=False,
            violation="rate_limit",
            message=(
                f"⚠️ You have reached the session limit of {_SESSION_LIMIT} queries. "
                "Please start a new session to continue."
            ),
        )
    return GuardrailResult(passed=True)


def reset_session(session_id: str) -> None:
    """Call this when a session ends or the user explicitly resets."""
    _session_counters.pop(session_id, None)


def get_session_count(session_id: str) -> int:
    return _session_counters.get(session_id, 0)


# ── 2. Prompt injection detection ───────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore\s+(your\s+)?instructions",
    r"forget\s+(your\s+)?(previous\s+|all\s+)?instructions",
    r"act\s+as(\s+a)?\s+(different|new|another)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"bypass\s+(rbac|restrictions?|security|access\s+control)",
    r"show\s+me\s+all\s+(documents?|files?|data)",
    r"regardless\s+of\s+(my\s+)?role",
    r"no\s+restrictions?",
    r"without\s+(any\s+)?restrictions?",
    r"override\s+(the\s+)?(system|prompt|instructions?)",
    r"system\s+prompt",
    r"jailbreak",
    r"developer\s+mode",
    r"\bdan\s+mode\b",
    r"ignore\s+previous\s+(message|context|prompt)",
    r"disregard\s+(your\s+)?(previous\s+)?instructions",
    r"new\s+persona",
    r"role\s*play\s+as",
    r"simulate\s+(being\s+)?a\s+(different|unrestricted)",
]]


def _check_prompt_injection(query: str) -> GuardrailResult:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                violation="prompt_injection",
                message=(
                    "🚫 Your query appears to contain an attempt to override system instructions "
                    "or bypass access controls. This action is not permitted and has been logged."
                ),
            )
    return GuardrailResult(passed=True)


# ── 3. PII – absolute blocks (any role) ─────────────────────────────────────

# Aadhar: 12 digits, optionally grouped as XXXX XXXX XXXX
_AADHAR_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
# PAN: AAAAA0000A format
_PAN_RE     = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
# Bank account: 9–18 consecutive digits
_BANK_RE    = re.compile(r"\b\d{9,18}\b")

_ABSOLUTE_BLOCK_CHECKS = [
    (_AADHAR_RE, "Aadhar number"),
    (_PAN_RE,    "PAN card number"),
    (_BANK_RE,   "bank account number"),
]


def _check_absolute_pii(query: str) -> GuardrailResult:
    for pattern, label in _ABSOLUTE_BLOCK_CHECKS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                violation="pii_absolute",
                message=(
                    f"🔒 Your query appears to contain a {label}. "
                    "Submission of financial identity numbers (Aadhar, PAN, bank accounts) "
                    "is strictly prohibited for all users. Please remove this information and retry."
                ),
            )
    return GuardrailResult(passed=True)


# ── 4. PII – HR-sensitive fields (only HR role may query) ───────────────────

_HR_SENSITIVE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bemployee[_\s-]?id\b",
    r"\bemp[_\s-]?id\b",
    r"\bdate\s+of\s+birth\b",
    r"\b(dob)\b",
    r"\bdate\s+of\s+joining\b",
    r"\b(doj)\b",
    r"\bmanager[_\s-]?id\b",
    r"\bsalary\b",
    r"\bctc\b",
    r"\bcompensation\s+(package|detail|breakdown)",
    r"\bleave\s+balance\b",
    r"\bleaves?\s+taken\b",
    r"\battendance\s+(record|detail|report)",
    r"\bperformance\s+rating\b",
    r"\blast\s+(performance\s+)?review\b",
    r"\bexit\s+date\b",
    r"\btermination\s+date\b",
]]


def _check_hr_sensitive_pii(query: str, user_access_roles: list[str]) -> GuardrailResult:
    if "HR" in user_access_roles:
        return GuardrailResult(passed=True)   # HR users may query these fields

    for pattern in _HR_SENSITIVE_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                violation="pii_hr_sensitive",
                message=(
                    "🔒 Your query references personal employee data "
                    "(e.g., salary, leave balance, date of birth, performance ratings). "
                    "Access to this information is restricted to HR personnel only."
                ),
            )
    return GuardrailResult(passed=True)


# ── 5. Off-topic detection ───────────────────────────────────────────────────

# Hard off-topic patterns (unambiguous non-business topics)
_OFFTOPIC_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bcricket\s+score\b",
    r"\bweather\s+(today|forecast|report)\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|song|story|essay|joke)\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\brecipe\s+for\b",
    r"\bwho\s+(won|is\s+winning)\s+the\s+(match|game|tournament)\b",
    r"\b(ipl|world\s+cup|fifa)\s+(score|result|winner)\b",
    r"\bcelebrity\s+(news|gossip)\b",
    r"\bhoroscope\b",
    r"\bstock\s+(market|price)\s+of\s+(?!finsolve)",   # stock prices (not FinSolve)
    r"\btranslate\s+.{1,50}\s+to\s+\w+\b",
    r"\bwhat\s+is\s+the\s+(capital|population)\s+of\b",
    r"\bdraw\s+(me\s+)?(a|an)\s+",
    r"\bgenerate\s+(an?\s+)?(image|picture|art)\b",
]]

# Minimum business-domain terms that count as "on-topic"
_BUSINESS_KEYWORDS: set[str] = {
    "budget", "revenue", "expense", "financial", "finance", "invoice", "vendor",
    "payroll", "compensation", "profit", "loss", "quarter", "annual", "fiscal",
    "sprint", "incident", "sla", "deployment", "infrastructure", "engineering",
    "system", "architecture", "uptime", "latency",
    "campaign", "marketing", "lead", "acquisition", "conversion", "brand",
    "employee", "leave", "leaves", "pto", "vacation", "holiday", "policy", "onboarding", "appraisal", "hr", "benefits",
    "strategy", "executive", "performance", "report", "forecast", "growth",
    "department", "company", "finsolve", "team", "project", "customer", "sales",
    "operations", "legal", "admin", "data", "technology",
}

_OFF_TOPIC_SCORE_THRESHOLD = 0.12   # lowered to prevent false-negative blocks for valid queries


def _check_off_topic(
    query: str,
    route_score: float | None = None,
) -> GuardrailResult:
    # Hard pattern match first
    for pattern in _OFFTOPIC_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                violation="off_topic",
                message=(
                    "😊 I'm sorry, but I can only assist with questions related to "
                    "FinSolve's business operations — Finance, Engineering, Marketing, HR, "
                    "or company strategy. Please rephrase your question accordingly."
                ),
            )

    # Semantic score check (when available from the router)
    if route_score is not None and route_score < _OFF_TOPIC_SCORE_THRESHOLD:
        return GuardrailResult(
            passed=False,
            violation="off_topic",
            message=(
                "😊 Your query doesn't appear to be related to FinSolve's business domains. "
                "I can help with Finance, Engineering, Marketing, HR, or executive strategy topics."
            ),
        )

    # Keyword fallback: if query has no business keywords and is nontrivial length
    lower = query.lower()
    words = set(re.findall(r"\b\w+\b", lower))
    if len(words) >= 4 and not (words & _BUSINESS_KEYWORDS):
        return GuardrailResult(
            passed=False,
            violation="off_topic",
            message=(
                "😊 Your query doesn't appear to be related to FinSolve's business domains. "
                "I can help with Finance, Engineering, Marketing, HR, or executive strategy topics."
            ),
        )

    return GuardrailResult(passed=True)


# ── Public input guardrail runner ────────────────────────────────────────────

class InputGuardrails:
    """
    Stateful input guardrail runner.  Maintains per-session query counters.

    Usage::

        guards = InputGuardrails()
        result = guards.check(query, session_id="user-123", user_access_roles=[...])
        if not result.passed:
            print(result.message)   # return this to the user
    """

    def __init__(self, session_limit: int = _SESSION_LIMIT):
        self._session_limit = session_limit
        self._counters: dict[str, int] = {}

    def check(
        self,
        query: str,
        session_id: str,
        user_access_roles: list[str],
        route_score: float | None = None,
    ) -> GuardrailResult:
        """
        Run all input guardrails in priority order and return the first failure.

        Args:
            query:             The raw user query string.
            session_id:        Opaque identifier for the user's session.
            user_access_roles: Roles from the HR data (e.g. ["C-Level", "General", "Finance"]).
            route_score:       Optional cosine-similarity score from the semantic router.
                               If provided, used to tighten off-topic detection.

        Returns:
            GuardrailResult — ``passed=True`` if all checks pass.
        """
        # 1. Rate limit
        count = self._counters.get(session_id, 0) + 1
        self._counters[session_id] = count
        if count > self._session_limit:
            return GuardrailResult(
                passed=False,
                violation="rate_limit",
                message=(
                    f"⚠️ You have reached the session limit of {self._session_limit} queries. "
                    "Please start a new session to continue."
                ),
            )

        # 2. Prompt injection
        result = _check_prompt_injection(query)
        if not result.passed:
            return result

        # 3. Absolute PII (any role)
        result = _check_absolute_pii(query)
        if not result.passed:
            return result

        # 4. HR-sensitive PII
        result = _check_hr_sensitive_pii(query, user_access_roles)
        if not result.passed:
            return result

        # 5. Off-topic
        result = _check_off_topic(query, route_score)
        if not result.passed:
            return result

        return GuardrailResult(passed=True, sanitized_query=query)

    def reset_session(self, session_id: str) -> None:
        self._counters.pop(session_id, None)

    def session_count(self, session_id: str) -> int:
        return self._counters.get(session_id, 0)


# ===========================================================================
# ── OUTPUT GUARDRAILS ───────────────────────────────────────────────────────
# ===========================================================================

# ── 1. Grounding check ───────────────────────────────────────────────────────

# Patterns for figures, dates, and currency amounts in free text
_FIGURE_RE = re.compile(
    r"(?:"
    r"₹\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakh|crore|million|billion|L|Cr|M|B))?"  # ₹ amounts
    r"|[\d,]{2,}(?:\.\d+)?%"                         # percentages
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"  # month year
    r"|q[1-4]\s*['\-]?\s*(?:fy)?\s*\d{2,4}"          # Q1 FY2024
    r"|fy\s*\d{4}"                                    # FY2024
    r"|(?:20|19)\d{2}"                               # 4-digit years
    r")",
    re.IGNORECASE,
)


def _check_grounding(
    response: str,
    retrieved_chunks: list[Any],
) -> tuple[bool, list[str]]:
    """
    Returns (is_grounded, list_of_ungrounded_figures).
    A figure is considered grounded if it appears verbatim in any retrieved chunk.
    """
    figures = _FIGURE_RE.findall(response)
    if not figures:
        return True, []

    chunk_corpus = " ".join(
        (c.text if hasattr(c, "text") else str(c)) for c in retrieved_chunks
    ).lower()

    ungrounded = [fig for fig in figures if fig.lower() not in chunk_corpus]
    return len(ungrounded) == 0, ungrounded


# ── 2. Cross-role leakage check ──────────────────────────────────────────────

# Terms that are strong indicators of content from a specific collection
_COLLECTION_MARKERS: dict[str, list[str]] = {
    "Finance": [
        "budget allocation", "accounts payable", "accounts receivable",
        "operating cost", "profit margin", "vendor payment", "capex", "opex",
        "fy2024", "fy2025", "financial summary", "department budget",
        "revenue target", "cost saving", "expenditure",
    ],
    "Engineering": [
        "sla breach", "mean time to recovery", "mttr", "p1 incident",
        "sprint velocity", "deployment frequency", "ci/cd", "uptime sla",
        "incident report", "on-call", "runbook",
    ],
    "Marketing": [
        "customer acquisition cost", "cost per lead", "conversion rate",
        "campaign roi", "lead generation", "brand awareness", "digital spend",
        "marketing budget", "channel performance",
    ],
    "HR": [
        # Note: "employee handbook" and "leave policy" are intentionally excluded
        # because the employee_handbook.pdf lives in the *general* collection and
        # is accessible to all users.  Only content that is truly HR-restricted
        # (e.g., personal attendance records, appraisals) should be flagged here.
        "leave balance", "performance appraisal",
        "grievance redressal", "onboarding checklist",
        "exit interview", "disciplinary action", "attendance record",
    ],
}


def _check_cross_role_leakage(
    response: str,
    user_access_roles: list[str],
) -> tuple[bool, list[str]]:
    """
    Returns (no_leakage, list_of_leaking_collections).
    Flags if the response contains marker terms from a collection the user can't access.
    """
    lower_response = response.lower()
    leaking = []
    for collection, markers in _COLLECTION_MARKERS.items():
        if collection in user_access_roles:
            continue                          # user has access → not a leak
        for marker in markers:
            if marker in lower_response:
                leaking.append(collection)
                break                         # one hit per collection is enough
    return len(leaking) == 0, leaking


# ── 3. Source citation check ──────────────────────────────────────────────────

_CITATION_RE = re.compile(
    r"(?:"
    r"source\s*[:—]"
    r"|according\s+to\s+(?:the\s+)?\w"
    r"|from\s+(?:the\s+)?\w+\.(?:pdf|docx|md)"
    r"|page\s+\d+"
    r"|p\.\s*\d+"
    r"|referenced\s+in"
    r"|see\s+\[?"
    r"|cited\s+in"
    r")",
    re.IGNORECASE,
)


def _check_source_citation(response: str) -> bool:
    """Returns True (citation present) or False (no citation detected)."""
    return bool(_CITATION_RE.search(response))


# ── Public output guardrail runner ────────────────────────────────────────────

def apply_output_guardrails(
    response: str,
    retrieved_chunks: list[Any],
    user_access_roles: list[str],
) -> OutputGuardrailResult:
    """
    Run all output guardrails against an LLM response before returning it to the user.

    Args:
        response:           The raw LLM response string.
        retrieved_chunks:   Chunk objects (or dicts) used as context for the LLM.
        user_access_roles:  Roles of the authenticated user.

    Returns:
        OutputGuardrailResult with (possibly annotated) response and any violation/warning flags.
    """
    violations: list[str] = []
    warnings: list[str]   = []
    annotated = response

    # 1. Grounding check
    is_grounded, ungrounded_figures = _check_grounding(response, retrieved_chunks)
    if not is_grounded:
        violations.append("ungrounded_content")
        disclaimer = (
            "\n\n⚠️ **Grounding Notice**: This response contains figures or dates "
            f"({', '.join(ungrounded_figures[:5])}) that could not be verified against "
            "the retrieved source documents. Please treat these with caution and cross-check "
            "with the original documents."
        )
        annotated += disclaimer

    # 2. Cross-role leakage check
    no_leakage, leaking_collections = _check_cross_role_leakage(response, user_access_roles)
    if not no_leakage:
        violations.append("cross_role_leakage")
        leak_warning = (
            f"\n\n🔒 **Access Notice**: This response may reference content from "
            f"{', '.join(leaking_collections)} — collection(s) you are not authorised to access. "
            "This has been flagged for review. The information should not be acted upon."
        )
        annotated += leak_warning

    # 3. Source citation check
    has_citation = _check_source_citation(response)
    if not has_citation:
        warnings.append("missing_citation")
        citation_warning = (
            "\n\n📌 **Citation Notice**: This response does not include explicit references "
            "to source documents or page numbers. For verification, please consult the "
            "original documents in the knowledge base."
        )
        annotated += citation_warning

    passed = len(violations) == 0
    return OutputGuardrailResult(
        passed=passed,
        violations=violations,
        warnings=warnings,
        response=annotated,
    )
