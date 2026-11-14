from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _ovec_estimators(random_state: int) -> list[tuple[str, Any]]:
    return [
        ("knn", KNeighborsClassifier()),
        ("rf", RandomForestClassifier(random_state=random_state, n_jobs=1)),
        ("logistic", LogisticRegression(max_iter=2000, random_state=random_state)),
        ("svm", SVC(probability=True, random_state=random_state)),
    ]


def _osec_estimators(random_state: int) -> list[tuple[str, Any]]:
    return [
        ("rf", RandomForestClassifier(random_state=random_state, n_jobs=1)),
        ("tree", DecisionTreeClassifier(random_state=random_state)),
        ("gradien", GradientBoostingClassifier(random_state=random_state)),
        ("bayes", GaussianNB()),
    ]


def build_ovec(
    *,
    weights: list[float] | None = None,
    voting: str = "hard",
    random_state: int = 42,
) -> VotingClassifier:
    """Build the released four-estimator OVEC shape.

    The source contains both weighted hard-voting and weighted soft-voting
    variants. Hyperparameter tuning is intentionally performed by the caller.
    """
    return VotingClassifier(
        estimators=_ovec_estimators(random_state),
        voting=voting,
        weights=weights,
        n_jobs=1,
    )


def build_osec(*, random_state: int = 42, cv: int = 5) -> StackingClassifier:
    """Build the base-paper OSEC shape with a logistic meta-classifier."""
    return StackingClassifier(
        estimators=_osec_estimators(random_state),
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        cv=cv,
        stack_method="auto",
        n_jobs=1,
    )
