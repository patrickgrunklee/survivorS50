"""Base model interface for Survivor ML.

All models (sklearn, Keras, custom) inherit from SurvivorModel.
This ensures compatibility with sklearn.base.clone and our CV framework.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SurvivorModel(BaseEstimator, RegressorMixin, ABC):
    """Abstract base for all Survivor ML models.

    Inherits from sklearn's BaseEstimator + RegressorMixin so that:
    - sklearn.base.clone() works automatically (uses get_params/set_params)
    - Can be used directly in run_cv() and evaluate_holdout()
    - Gets score() for free from RegressorMixin

    Subclasses must implement:
        fit(X, y) -> self
        predict(X) -> np.ndarray
    """

    @abstractmethod
    def fit(self, X, y) -> 'SurvivorModel':
        """Train the model on features X and target y."""
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict placements for features X."""
        ...

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Short description of the model."""
        return f"{self.name} with params: {self.get_params()}"
