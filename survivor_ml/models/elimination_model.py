"""LightGBM ranking models for tribal council elimination prediction.

Each tribal council is a query group and the players attending it are
documents to rank.  The model learns which player is most likely to be
eliminated (relevance 0) vs survive (relevance 1).

Two pre-configured variants:
    EliminationLambdaRank  — ``objective='lambdarank'``
    EliminationPairwise    — ``objective='rank_xendcg'`` (cross-entropy pairwise)

Both share the same ``EliminationRankModel`` class, just different objectives.

Usage::

    from survivor_ml.elimination_features import (
        build_returnee_seasons_dataset,
        get_elimination_feature_columns,
        get_query_groups,
    )
    from survivor_ml.models.elimination_model import EliminationRankModel

    df = build_returnee_seasons_dataset()
    X = df[get_elimination_feature_columns()].values
    y = df["relevance"].values
    group = get_query_groups(df)

    model = EliminationRankModel(objective="lambdarank")
    model.fit(X, y, group=group)

    # Raw survival scores (higher = more likely to survive)
    scores = model.predict(X)

    # Elimination probabilities per tribal (sum to 1 within each group)
    probs = model.predict_elimination_probs(X, group=group)
"""

import numpy as np

try:
    import lightgbm as lgb

    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

from .base import SurvivorModel


class EliminationRankModel(SurvivorModel):
    """LightGBM learning-to-rank model for elimination prediction.

    Wraps LightGBM's built-in ranking objectives.  Follows the SurvivorModel
    pattern (fit / predict / get_params / set_params) with an extra *group*
    parameter on ``fit`` for query-group sizes.

    Parameters
    ----------
    objective : str
        LightGBM ranking objective.  ``'lambdarank'`` (default) or
        ``'rank_xendcg'`` (cross-entropy pairwise).
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Step-size shrinkage.
    num_leaves : int
        Max leaves per tree.
    min_data_in_leaf : int
        Minimum samples in a leaf.
    label_gain : list[float] | None
        Gain per relevance level.  Default ``[0, 1]`` for binary survived/eliminated.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        objective: str = "lambdarank",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_data_in_leaf: int = 5,
        label_gain: list[float] | None = None,
        random_state: int = 42,
    ):
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.label_gain = label_gain
        self.random_state = random_state

    # ------------------------------------------------------------------
    # SurvivorModel interface
    # ------------------------------------------------------------------

    def fit(self, X, y, group=None) -> "EliminationRankModel":
        """Train the ranking model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix — rows ordered so that rows within the same query
            group are contiguous.
        y : array-like of shape (n_samples,)
            Relevance labels (0 = eliminated, 1 = survived).
        group : array-like of int, optional
            Number of rows in each query group.  Required for ranking.
            If ``None``, all rows are treated as a single group.

        Returns
        -------
        self
        """
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is required for EliminationRankModel. "
                "Install it with: pip install lightgbm"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if group is None:
            group = [len(y)]

        params = {
            "objective": self.objective,
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "seed": self.random_state,
            "verbose": -1,
        }
        if self.label_gain is not None:
            params["label_gain"] = self.label_gain
        else:
            # Binary relevance: eliminated=0 (gain 0), survived=1 (gain 1)
            params["label_gain"] = [0.0, 1.0]

        train_ds = lgb.Dataset(X, label=y, group=list(group))
        self._booster = lgb.train(
            params, train_ds, num_boost_round=self.n_estimators,
        )
        self._n_features = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        """Return raw ranking scores (higher = more likely to survive)."""
        X = np.asarray(X, dtype=np.float64)
        return self._booster.predict(X)

    # ------------------------------------------------------------------
    # Elimination-specific methods
    # ------------------------------------------------------------------

    def predict_elimination_probs(
        self,
        X,
        group=None,
    ) -> np.ndarray:
        """Predict per-player elimination probability.

        Within each query group, applies a softmin over survival scores so
        the player with the *lowest* survival score gets the *highest*
        elimination probability.  Probabilities sum to 1 within each group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        group : array-like of int, optional
            Group sizes.  If ``None``, treats all rows as one group.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Elimination probability per player.
        """
        scores = self.predict(X)

        if group is None:
            group = [len(scores)]

        probs = np.empty_like(scores)
        idx = 0
        for g in group:
            s = scores[idx : idx + g]
            # Softmin: negate scores so lower survival → higher probability
            neg = -s
            neg -= neg.max()  # numerical stability
            exp_neg = np.exp(neg)
            probs[idx : idx + g] = exp_neg / exp_neg.sum()
            idx += g

        return probs

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Return feature importance from the trained booster."""
        return self._booster.feature_importance(importance_type=importance_type)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        obj_label = {
            "lambdarank": "LambdaRank",
            "rank_xendcg": "Pairwise",
        }.get(self.objective, self.objective)
        return f"Elimination{obj_label}"

    def score(self, X, y, group=None) -> float:
        """Mean NDCG@1 across query groups.

        For each tribal council, checks whether the model's top-ranked
        (highest survival score) player actually survived.
        """
        if group is None:
            return 0.0

        scores = self.predict(X)
        y = np.asarray(y)

        idx = 0
        correct = 0
        n_groups = len(group)
        for g in group:
            gs = scores[idx : idx + g]
            gy = y[idx : idx + g]
            # Lowest-ranked player should be the eliminated one (label 0).
            # Equivalently: highest-ranked player should have label 1.
            if gy[np.argmax(gs)] == 1:
                correct += 1
            idx += g

        return correct / max(n_groups, 1)


# ------------------------------------------------------------------
# Pre-configured model variants
# ------------------------------------------------------------------

def get_all_elimination_models() -> dict[str, EliminationRankModel]:
    """Return pre-configured elimination model variants."""
    if not _LIGHTGBM_AVAILABLE:
        return {}
    return {
        "EliminationLambdaRank": EliminationRankModel(
            objective="lambdarank",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=5,
        ),
        "EliminationPairwise": EliminationRankModel(
            objective="rank_xendcg",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=5,
        ),
    }
