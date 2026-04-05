"""
app/semantic_routing.py
Builds 5 semantic routes and classifies incoming queries into the appropriate domain.

Routes: Finance | Engineering | Marketing | HR | C-Level

Each route is defined by representative utterances.  At build time the router
computes a single centroid embedding (mean of utterance embeddings) per route.
At query time it picks the route whose centroid is closest to the query vector.

Access control:
    The router checks whether the user's access_roles permit the matched route.
    Users with a common role (C-Level or General) are always allowed through —
    the Qdrant RBAC filter will scope results to only what they can see.
    Restricted department routes (Finance, Engineering, Marketing, HR) are only
    blocked when the user has no department-level overlap AND no common role that
    maps to a collection visible to them.
"""

from dataclasses import dataclass

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer

from app.config import settings

# Roles that grant access to general/cross-department documents.
# Users holding any of these roles are always permitted past the route gate;
# Qdrant's payload filter will restrict results to what they can actually see.
_COMMON_ROLES: frozenset[str] = frozenset(settings.chunk_common_roles)

# ---------------------------------------------------------------------------
# Route utterances
# Each list contains domain-representative questions/phrases.
# Add or remove utterances to tune routing accuracy.
# ---------------------------------------------------------------------------

ROUTE_UTTERANCES: dict[str, list[str]] = {
    "Finance": [
        "What is the quarterly revenue?",
        "Show me the annual budget breakdown by department",
        "What were the total vendor payments this quarter?",
        "Explain the variance between actual and projected expenses",
        "What is the financial forecast for next fiscal year?",
        "How did operating costs change year over year?",
        "What is the profit margin for Q3?",
        "List all accounts payable outstanding",
        "What is the RPA automation cost savings?",
        "Summarise the department budget for technology",
    ],
    "Engineering": [
        "What are the SLA metrics for our production systems?",
        "Show me the incident report from last quarter",
        "What is the current sprint velocity?",
        "Explain the system architecture and tech stack",
        "How many P1 incidents occurred this year?",
        "What is the mean time to recovery for outages?",
        "List the open engineering tickets and their status",
        "What infrastructure changes were made in Q4?",
        "Describe the CI/CD pipeline we use",
        "What are the engineering OKRs for this quarter?",
    ],
    "Marketing": [
        "What was the Q1 marketing campaign performance?",
        "How much did we spend on customer acquisition?",
        "Which campaigns generated the most leads?",
        "What is our brand positioning strategy?",
        "Show me the digital marketing ROI",
        "What is the customer lifetime value?",
        "Summarise the quarterly marketing report",
        "What channels drove the highest conversion rate?",
        "Describe our content marketing strategy",
        "What is the target audience for our product?",
    ],
    "HR": [
        "How many leaves do I get?",
        "What is my PTO or vacation balance limit?",
        "What is the company leave policy?",
        "How does the employee onboarding process work?",
        "What health and wellness benefits are available?",
        "Explain the performance review and appraisal cycle",
        "What is the policy on remote and hybrid work?",
        "How do I apply for a promotion?",
        "What is the code of conduct and disciplinary process?",
        "Describe employee training and development programmes",
        "What is the grievance redressal procedure?",
        "What are the rules around expense reimbursement?",
    ],
    "C-Level": [
        "Give me an executive summary of company performance",
        "What are the strategic priorities for the next fiscal year?",
        "How is the company tracking against its annual targets?",
        "What are the top business risks and mitigation plans?",
        "Summarise cross-departmental budget versus actuals",
        "What is the company headcount plan for next year?",
        "How are we positioned relative to our competitors?",
        "What major decisions were made in the last board meeting?",
        "What is the overall revenue growth trajectory?",
        "Explain the company expansion strategy into new markets",
    ],
    "General": [
        "What are the company's core values?",
        "Where is the main office located?",
        "What is the dress code policy?",
        "How do I access the employee portal?",
        "What are the company holidays?",
        "Who is the CEO?",
        "Explain our mission statement",
        "What is our corporate social responsibility policy?",
    ],
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RouteMatch:
    route: str           # matched route name e.g. "Finance"
    score: float         # cosine similarity score (0–1, higher = more confident)
    allowed: bool        # True if the user's access_roles include this route


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------

class SemanticRouter:
    """
    Classifies a query into one of the 5 department routes using cosine
    similarity against pre-computed route centroid embeddings.
    """

    def __init__(
        self,
        utterances: dict[str, list[str]] = ROUTE_UTTERANCES,
        model_name: str = settings.embedding_model,
    ):
        print(f"Building semantic router with model '{model_name}'…")
        try:
            self._model = SentenceTransformer(model_name)
        except Exception:
            print("Network unavailable, loading router model from local cache…")
            self._model = SentenceTransformer(model_name, local_files_only=True)
        self._centroids: dict[str, np.ndarray] = {}
        self._build(utterances)

    def _build(self, utterances: dict[str, list[str]]) -> None:
        """Encode all utterances and store one mean centroid per route."""
        for route, texts in utterances.items():
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Mean centroid, then re-normalise so cosine sim stays in [0, 1]
            centroid = embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            self._centroids[route] = centroid / norm if norm > 0 else centroid
            print(f"  Route '{route}' — {len(texts)} utterances encoded.")
        print("Semantic router ready.\n")

    def route(self, query: str, user_access_roles: list[str]) -> RouteMatch:
        """
        Classify a query into the best-matching route and check user permission.

        Args:
            query:             The user's natural-language question.
            user_access_roles: Roles assigned to the authenticated user
                               (from hr_loader / employees.csv).

        Returns:
            RouteMatch with the matched route, similarity score, and access flag.

        Access logic:
            - If the user has the matched route in their roles → allowed.
            - If the user has ANY common role (C-Level / General) → allowed.
              Documents in general-access collections are visible to everyone;
              Qdrant's RBAC filter will restrict results appropriately.
            - Otherwise → denied (user has no overlap with the matched department
              AND no common-access privilege).
        """
        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        best_route, best_score = "", -1.0
        for route, centroid in self._centroids.items():
            score = float(np.dot(q_vec, centroid))
            if score > best_score:
                best_score = score
                best_route = route

        roles_set = set(user_access_roles)

        # Allow if the user has the specific department role OR any common role.
        # Common-role users (General / C-Level) can ask about any topic —
        # Qdrant will return only the chunks their roles actually permit.
        allowed = (
            best_route in roles_set
            or bool(roles_set & _COMMON_ROLES)
        )

        return RouteMatch(route=best_route, score=round(best_score, 4), allowed=allowed)


# ---------------------------------------------------------------------------
# Qdrant access-control filter
# ---------------------------------------------------------------------------

def get_qdrant_filter(user_access_roles: list[str]) -> Filter:
    """
    Build a Qdrant Filter that restricts search results to points whose
    `access_roles` payload field contains at least one of the user's roles.

    Usage:
        results = client.search(
            collection_name=...,
            query_vector=...,
            query_filter=get_qdrant_filter(user_roles),
        )
    """
    return Filter(
        must=[
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=user_access_roles),
            )
        ]
    )


# ---------------------------------------------------------------------------
# Singleton builder
# ---------------------------------------------------------------------------

_router: SemanticRouter | None = None


def build_router() -> SemanticRouter:
    """Return the shared SemanticRouter, building it on first call."""
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router
