"""
app/quality/suitability_model.py

Thin wrapper around the trained Logistic Regression Suitability Model
(models/suitability_logreg.joblib, produced by
app/training/train_suitability_model.py). Loads once, predicts many.
"""

import logging

import joblib
import pandas as pd

from app.quality.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)


class SuitabilityModel:
    """
    Loads the trained Suitability classifier and exposes a simple
    predict(feature_vector) -> (label, confidence) interface.
    """

    def __init__(self, model_path: str = "models/suitability_logreg.joblib"):
        logger.info(f"Loading Suitability Model from {model_path}...")
        self.model = joblib.load(model_path)
        logger.info("Suitability Model loaded.")

    def predict(self, feature_vector: list[float]) -> tuple[str, float]:
        """
        feature_vector: 8 normalized 0-1 scores, in FEATURE_NAMES order
        (as returned by FeatureExtractor.extract_vector()).

        Returns (label, confidence) where label is "Suitable" or
        "Not Suitable", and confidence is the model's predicted
        probability of that label (0.0-1.0).
        """
        # Wrapped in a DataFrame with the same column names used during
        # training (see train_suitability_model.py) -- avoids a sklearn
        # UserWarning about missing feature names, and guarantees column
        # order matches what the model was fitted on.
        X = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)

        label = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # predict_proba's column order matches self.model.classes_
        class_index = list(self.model.classes_).index(label)
        confidence = float(probabilities[class_index])

        return label, confidence
