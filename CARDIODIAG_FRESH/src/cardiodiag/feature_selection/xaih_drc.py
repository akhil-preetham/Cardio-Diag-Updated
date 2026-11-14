from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class XAIHDRCUnderSpecifiedError(NotImplementedError):
    """Raised when paper-level XAIH-DRC parameters are unavailable."""


class XAIHDRCSelector(BaseEstimator, TransformerMixin):
    """Reproduce the released NXAIH-DRC correlation/LIME/PCA procedure.

    The notebooks define top-k thresholds (k=11), retain features that pass
    both absolute correlation and global LIME-importance thresholds, and
    project XOR-disagreement features into nine PCA components. LIME
    importances must be computed from the training fold and supplied here;
    silently computing them from the full dataset would leak information.
    """

    def __init__(self, *, n_features: int = 11, pca_components: int = 9):
        self.n_features = n_features
        self.pca_components = pca_components

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        lime_importances: dict[str, float] | pd.Series | None = None,
    ):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if lime_importances is None:
            raise XAIHDRCUnderSpecifiedError(
                "Training-fold global LIME importances are required for XAIH-DRC."
            )
        numeric = X.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().all(axis=None):
            raise ValueError("XAIH-DRC requires at least one numeric feature")
        target = pd.Series(y, index=X.index).astype(float)
        correlations = numeric.corrwith(target).abs().fillna(0.0)
        self.feature_scores_ = correlations.to_dict()
        importance = pd.Series(lime_importances, dtype=float).reindex(numeric.columns).fillna(0.0)
        self.lime_scores_ = importance.to_dict()
        corr_threshold = correlations.sort_values(ascending=False).iloc[min(self.n_features, len(correlations)) - 1]
        lime_threshold = importance.sort_values(ascending=False).iloc[min(self.n_features, len(importance)) - 1]
        corr_keep = correlations >= corr_threshold
        lime_keep = importance >= lime_threshold
        self.selected_features_ = list(numeric.columns[corr_keep & lime_keep])
        self.pca_features_ = list(numeric.columns[corr_keep ^ lime_keep])
        self.pca_ = None
        if self.pca_features_:
            components = min(self.pca_components, len(self.pca_features_), len(numeric))
            self.pca_ = PCA(n_components=components).fit(numeric[self.pca_features_])
        if not self.selected_features_ and self.pca_ is None:
            raise ValueError("XAIH-DRC selected no usable features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("XAIHDRCSelector must be fitted before transform")
        required = set(self.selected_features_) | set(self.pca_features_)
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Selected features are missing from X: {missing}")
        parts = [X.loc[:, self.selected_features_]] if self.selected_features_ else []
        if self.pca_ is not None:
            transformed = self.pca_.transform(X.loc[:, self.pca_features_])
            parts.append(
                pd.DataFrame(
                    transformed,
                    columns=[f"PCA_{index + 1}" for index in range(transformed.shape[1])],
                    index=X.index,
                )
            )
        return pd.concat(parts, axis=1)
