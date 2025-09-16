"""
Regime detection modules for portfolio optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger(__name__)


class RegimeDetector:
    """Base class for regime detection."""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_labels = None
        self.regime_probs = None
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame) -> None:
        """Fit the regime detection model."""
        raise NotImplementedError
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes for given features."""
        raise NotImplementedError
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for given features."""
        raise NotImplementedError
    
    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        """Fit the model and predict regimes in one step."""
        self.fit(features)
        return self.predict(features)
    
    def get_regime_characteristics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get characteristics of each regime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            regime_mask = self.regime_labels == regime
            regime_features = features[regime_mask]
            
            if len(regime_features) > 0:
                regime_stats[f"regime_{regime}"] = {
                    "count": len(regime_features),
                    "percentage": len(regime_features) / len(features) * 100,
                    "mean": regime_features.mean().to_dict(),
                    "std": regime_features.std().to_dict()
                }
        
        return regime_stats


class HMMRegimeDetector(RegimeDetector):
    """Hidden Markov Model regime detector."""
    
    def __init__(self, n_regimes: int = 3, n_iter: int = 100, random_state: int = 42):
        super().__init__(n_regimes)
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, features: pd.DataFrame) -> None:
        """Fit HMM regime detection model."""
        
        logger.info("Fitting HMM regime detector", n_regimes=self.n_regimes, n_features=len(features.columns))
        
        # Prepare features
        X = features.fillna(method='ffill').fillna(method='bfill').values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit HMM model
        from hmmlearn import hmm
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled)
        
        # Predict regimes
        self.regime_labels = self.model.predict(X_scaled)
        self.regime_probs = self.model.predict_proba(X_scaled)
        self.is_fitted = True
        
        logger.info("HMM regime detector fitted successfully", n_regimes=self.n_regimes)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes for given features."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.fillna(method='ffill').fillna(method='bfill').values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for given features."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.fillna(method='ffill').fillna(method='bfill').values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition matrix."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.transmat_
    
    def get_regime_means(self) -> np.ndarray:
        """Get regime means."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.means_
    
    def get_regime_covariances(self) -> np.ndarray:
        """Get regime covariances."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.covars_


class LSTMRegimeDetector(RegimeDetector):
    """LSTM-based regime detector."""
    
    def __init__(
        self,
        n_regimes: int = 3,
        sequence_length: int = 60,
        hidden_units: int = 64,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42
    ):
        super().__init__(n_regimes)
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, features: pd.DataFrame) -> None:
        """Fit LSTM regime detection model."""
        
        logger.info("Fitting LSTM regime detector", n_regimes=self.n_regimes, n_features=len(features.columns))
        
        # Prepare features
        X = features.fillna(method='ffill').fillna(method='bfill').values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled)
        
        # Convert to categorical
        y_categorical = self._to_categorical(y_seq)
        
        # Build LSTM model
        self.model = self._build_model(X_seq.shape[1], X_seq.shape[2], self.n_regimes)
        
        # Train model
        self.model.fit(
            X_seq, y_categorical,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Predict regimes
        self.regime_labels = self.model.predict(X_seq).argmax(axis=1)
        self.regime_probs = self.model.predict(X_seq)
        self.is_fitted = True
        
        logger.info("LSTM regime detector fitted successfully", n_regimes=self.n_regimes)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes for given features."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.fillna(method='ffill').fillna(method='bfill').values
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_seq)
        
        return predictions.argmax(axis=1)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for given features."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.fillna(method='ffill').fillna(method='bfill').values
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Predict probabilities
        return self.model.predict(X_seq)
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(i)  # Use index as target for now
        
        return np.array(X), np.array(y)
    
    def _to_categorical(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to categorical."""
        
        from tensorflow.keras.utils import to_categorical
        
        # For now, create dummy labels
        # In practice, you would use actual regime labels
        dummy_labels = np.random.randint(0, self.n_regimes, len(y))
        
        return to_categorical(dummy_labels, num_classes=self.n_regimes)
    
    def _build_model(self, sequence_length: int, n_features: int, n_regimes: int):
        """Build LSTM model."""
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(self.hidden_units, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(self.dropout),
            LSTM(self.hidden_units, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(n_regimes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class GMMRegimeDetector(RegimeDetector):
    """Gaussian Mixture Model regime detector."""
    
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        super().__init__(n_regimes)
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, features: pd.DataFrame) -> None:
        """Fit GMM regime detection model."""
        
        logger.info("Fitting GMM regime detector", n_regimes=self.n_regimes, n_features=len(features.columns))
        
        # Prepare features
        X = features.fillna(method='ffill').fillna(method='bfill').values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit GMM model
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled)
        
        # Predict regimes
        self.regime_labels = self.model.predict(X_scaled)
        self.regime_probs = self.model.predict_proba(X_scaled)
        self.is_fitted = True
        
        logger.info("GMM regime detector fitted successfully", n_regimes=self.n_regimes)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes for given features."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.fillna(method='ffill').fillna(method='bfill').values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for given features."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.fillna(method='ffill').fillna(method='bfill').values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def get_regime_means(self) -> np.ndarray:
        """Get regime means."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.means_
    
    def get_regime_covariances(self) -> np.ndarray:
        """Get regime covariances."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.covariances_
    
    def get_regime_weights(self) -> np.ndarray:
        """Get regime weights."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.weights_


class RegimeDetector:
    """Main regime detector class that can use different methods."""
    
    def __init__(
        self,
        method: str = "hmm",
        n_regimes: int = 3,
        **kwargs
    ):
        self.method = method
        self.n_regimes = n_regimes
        
        # Initialize detector based on method
        if method == "hmm":
            self.detector = HMMRegimeDetector(n_regimes=n_regimes, **kwargs)
        elif method == "lstm":
            self.detector = LSTMRegimeDetector(n_regimes=n_regimes, **kwargs)
        elif method == "gmm":
            self.detector = GMMRegimeDetector(n_regimes=n_regimes, **kwargs)
        else:
            raise ValueError(f"Unknown regime detection method: {method}")
    
    def fit(self, features: pd.DataFrame) -> None:
        """Fit the regime detection model."""
        self.detector.fit(features)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes for given features."""
        return self.detector.predict(features)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for given features."""
        return self.detector.predict_proba(features)
    
    def get_regime_characteristics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get characteristics of each regime."""
        return self.detector.get_regime_characteristics(features)
    
    def get_regime_labels(self) -> np.ndarray:
        """Get regime labels from training."""
        return self.detector.regime_labels
    
    def get_regime_probs(self) -> np.ndarray:
        """Get regime probabilities from training."""
        return self.detector.regime_probs
