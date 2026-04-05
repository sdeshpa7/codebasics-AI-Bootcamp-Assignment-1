"""
app/semantic_routing.py
Builds 5 semantic routes and classifies incoming queries into the appropriate domain.

Routes: Finance | Engineering | Marketing | HR | C-Level | General

Access control:
    - If best_route is "General" -> Allowed for all.
    - If best_route matches a department -> Only allowed if user has that specific role or C-Level.
"""

from dataclasses import dataclass
import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer
from app.config import settings

# Only C-Level has global bypass across all categories.
_COMMON_ROLES: frozenset[str] = frozenset(settings.chunk_common_roles)

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
        "How does the employee onboarding process work?",
        "Explain the performance review and appraisal cycle",
        "How do I apply for a promotion?",
        "What is the grievance redressal procedure?",
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
        "How many leaves do I get?",
        "What is my PTO or vacation balance limit?",
        "What is the company leave policy?",
        "What health and wellness benefits are available?",
        "What is the policy on remote and hybrid work?",
        "What is the code of conduct and disciplinary process?",
        "Describe employee training and development programmes",
        "What are the rules around expense reimbursement?",
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

@dataclass
class RouteMatch:
    route: str
    score: float
    allowed: bool

class SemanticRouter:
    def __init__(
        self,
        utterances: dict[str, list[str]] = ROUTE_UTTERANCES,
        model_name: str = settings.embedding_model,
    ):
        print(f"Building semantic router with model '{model_name}'…")
        try:
            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        self._centroids: dict[str, np.ndarray] = {}
        self._build(utterances)

    def _build(self, utterances: dict[str, list[str]]) -> None:
        for route, texts in utterances.items():
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            centroid = embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            self._centroids[route] = centroid / norm if norm > 0 else centroid

    def route(self, query: str, user_access_roles: list[str]) -> RouteMatch:
        q_vec = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        best_route, best_score = "", -1.0
        for route, centroid in self._centroids.items():
            score = float(np.dot(q_vec, centroid))
            if score > best_score:
                best_score = score
                best_route = route

        roles_set = set(user_access_roles)
        
        # Access Logic:
        # 1. User has the specific department role -> Allowed.
        # 2. Matched route is "General" (Handbook) -> Allowed.
        # 3. User is C-Level -> Allowed.
        allowed = (
            best_route in roles_set
            or best_route == "General"
            or bool(roles_set & _COMMON_ROLES)
        )

        return RouteMatch(route=best_route, score=round(best_score, 4), allowed=allowed)

def get_qdrant_filter(user_access_roles: list[str]) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="access_roles", match=MatchAny(any=user_access_roles))
        ]
    )

_router: SemanticRouter | None = None

def build_router() -> SemanticRouter:
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router
