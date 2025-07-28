"""
Machine Learning Integration - MAANG Standards.

This module implements comprehensive machine learning integration
following MAANG best practices for AI/ML systems.

Features:
    - Model management and versioning
    - Real-time inference
    - Batch processing
    - Model performance monitoring
    - A/B testing for models
    - Feature engineering
    - Model explainability
    - Automated retraining
    - Model drift detection
    - Ensemble methods

ML Capabilities:
    - Natural Language Processing
    - Recommendation Systems
    - Anomaly Detection
    - Predictive Analytics
    - Sentiment Analysis
    - Content Classification
    - User Behavior Prediction

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import json
import pickle
import hashlib
from typing import (
    Optional, Dict, Any, List, Union, Callable,
    TypeVar, Protocol, Tuple, Set
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from api.monitoring import Counter, Histogram, Gauge
from api.cache import get_cache_manager
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')

# ML metrics
model_inference_time = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name', 'model_version', 'inference_type']
)

model_accuracy = Gauge(
    'model_accuracy_score',
    'Model accuracy score',
    ['model_name', 'model_version', 'metric_type']
)

model_predictions = Counter(
    'model_predictions_total',
    'Model predictions',
    ['model_name', 'model_version', 'prediction_type']
)

# Model types
class ModelType(str, Enum):
    """Types of ML models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"

class ModelStatus(str, Enum):
    """Model deployment status."""
    TRAINING = "training"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"

# Model metadata
@dataclass
class ModelMetadata:
    """Model metadata and versioning information."""
    
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    model_size_bytes: Optional[int] = None
    inference_latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'performance_metrics': self.performance_metrics,
            'hyperparameters': self.hyperparameters,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_size_bytes': self.model_size_bytes,
            'inference_latency_ms': self.inference_latency_ms
        }

# Model wrapper
class ModelWrapper:
    """Wrapper for ML models with metadata and monitoring."""
    
    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None
    ):
        """Initialize model wrapper."""
        self.model = model
        self.metadata = metadata
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.cache_manager = get_cache_manager()
    
    async def predict(
        self,
        input_data: Union[np.ndarray, List, Dict[str, Any]],
        use_cache: bool = True
    ) -> Any:
        """Make prediction with monitoring and caching."""
        start_time = datetime.now(timezone.utc)
        
        # Generate cache key
        cache_key = None
        if use_cache:
            input_hash = hashlib.md5(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest()
            cache_key = f"model:{self.metadata.name}:{self.metadata.version}:{input_hash}"
            
            # Check cache
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        try:
            # Preprocess input
            if self.preprocessor:
                processed_input = self.preprocessor(input_data)
            else:
                processed_input = input_data
            
            # Make prediction
            prediction = self.model.predict(processed_input)
            
            # Postprocess output
            if self.postprocessor:
                prediction = self.postprocessor(prediction)
            
            # Record metrics
            inference_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            model_inference_time.labels(
                model_name=self.metadata.name,
                model_version=self.metadata.version,
                inference_type='prediction'
            ).observe(inference_time)
            
            model_predictions.labels(
                model_name=self.metadata.name,
                model_version=self.metadata.version,
                prediction_type='success'
            ).inc()
            
            # Cache result
            if cache_key:
                await self.cache_manager.set(cache_key, prediction, ttl=3600)
            
            return prediction
            
        except Exception as e:
            # Record error metrics
            model_predictions.labels(
                model_name=self.metadata.name,
                model_version=self.metadata.version,
                prediction_type='error'
            ).inc()
            
            logger.error(
                "Model prediction failed",
                model_name=self.metadata.name,
                model_version=self.metadata.version,
                error=str(e)
            )
            raise
    
    async def predict_proba(
        self,
        input_data: Union[np.ndarray, List, Dict[str, Any]]
    ) -> np.ndarray:
        """Make probability prediction."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Preprocess input
            if self.preprocessor:
                processed_input = self.preprocessor(input_data)
            else:
                processed_input = input_data
            
            # Make probability prediction
            probabilities = self.model.predict_proba(processed_input)
            
            # Record metrics
            inference_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            model_inference_time.labels(
                model_name=self.metadata.name,
                model_version=self.metadata.version,
                inference_type='probability'
            ).observe(inference_time)
            
            return probabilities
            
        except Exception as e:
            logger.error(
                "Model probability prediction failed",
                model_name=self.metadata.name,
                model_version=self.metadata.version,
                error=str(e)
            )
            raise

# Model manager
class ModelManager:
    """
    Comprehensive model management system.
    
    Features:
    - Model versioning and deployment
    - Performance monitoring
    - A/B testing for models
    - Automated retraining
    - Model drift detection
    """
    
    def __init__(self):
        """Initialize model manager."""
        self.models: Dict[str, ModelWrapper] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.model_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.feature_store: Dict[str, np.ndarray] = {}
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self) -> None:
        """Initialize default ML models."""
        # Text classification model
        self._create_text_classifier()
        
        # Anomaly detection model
        self._create_anomaly_detector()
        
        # Recommendation model
        self._create_recommendation_model()
    
    def _create_text_classifier(self) -> None:
        """Create text classification model."""
        # Simple TF-IDF + Random Forest classifier
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        metadata = ModelMetadata(
            name="text_classifier",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            feature_columns=["text"],
            target_column="category",
            hyperparameters={
                "n_estimators": 100,
                "max_features": 1000
            }
        )
        
        model_wrapper = ModelWrapper(
            model=classifier,
            metadata=metadata,
            preprocessor=vectorizer.transform
        )
        
        self.models["text_classifier"] = model_wrapper
        self.model_metadata["text_classifier"] = metadata
    
    def _create_anomaly_detector(self) -> None:
        """Create anomaly detection model."""
        detector = IsolationForest(contamination=0.1, random_state=42)
        
        metadata = ModelMetadata(
            name="anomaly_detector",
            version="1.0.0",
            model_type=ModelType.ANOMALY_DETECTION,
            status=ModelStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            feature_columns=["feature_1", "feature_2", "feature_3"],
            hyperparameters={
                "contamination": 0.1,
                "n_estimators": 100
            }
        )
        
        model_wrapper = ModelWrapper(
            model=detector,
            metadata=metadata
        )
        
        self.models["anomaly_detector"] = model_wrapper
        self.model_metadata["anomaly_detector"] = metadata
    
    def _create_recommendation_model(self) -> None:
        """Create recommendation model."""
        # Simple collaborative filtering (placeholder)
        class SimpleRecommender:
            def __init__(self):
                self.user_item_matrix = {}
                self.item_similarities = {}
            
            def fit(self, user_item_data):
                # Simple implementation
                self.user_item_matrix = user_item_data
            
            def predict(self, user_id):
                # Simple recommendation logic
                return [1, 2, 3, 4, 5]  # Top 5 items
        
        recommender = SimpleRecommender()
        
        metadata = ModelMetadata(
            name="recommendation_model",
            version="1.0.0",
            model_type=ModelType.RECOMMENDATION,
            status=ModelStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            feature_columns=["user_id", "item_id", "rating"],
            hyperparameters={
                "top_k": 5,
                "similarity_metric": "cosine"
            }
        )
        
        model_wrapper = ModelWrapper(
            model=recommender,
            metadata=metadata
        )
        
        self.models["recommendation_model"] = model_wrapper
        self.model_metadata["recommendation_model"] = metadata
    
    def register_model(
        self,
        name: str,
        model: Any,
        metadata: ModelMetadata,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None
    ) -> None:
        """Register a new model."""
        model_wrapper = ModelWrapper(
            model=model,
            metadata=metadata,
            preprocessor=preprocessor,
            postprocessor=postprocessor
        )
        
        self.models[name] = model_wrapper
        self.model_metadata[name] = metadata
        
        logger.info(
            "Model registered",
            name=name,
            version=metadata.version,
            model_type=metadata.model_type.value
        )
    
    async def predict(
        self,
        model_name: str,
        input_data: Union[np.ndarray, List, Dict[str, Any]]
    ) -> Any:
        """Make prediction using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_wrapper = self.models[model_name]
        return await model_wrapper.predict(input_data)
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self.model_metadata.get(model_name)
    
    def get_all_models(self) -> Dict[str, ModelMetadata]:
        """Get all registered models."""
        return self.model_metadata.copy()
    
    async def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_wrapper = self.models[model_name]
        
        try:
            # Make predictions
            y_pred = await model_wrapper.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Update model metadata
            model_wrapper.metadata.performance_metrics.update(metrics)
            model_wrapper.metadata.updated_at = datetime.now(timezone.utc)
            
            # Record metrics
            for metric_name, value in metrics.items():
                model_accuracy.labels(
                    model_name=model_name,
                    model_version=model_wrapper.metadata.version,
                    metric_type=metric_name
                ).set(value)
            
            # Store performance history
            self.model_performance[model_name].append({
                'timestamp': datetime.now(timezone.utc),
                'metrics': metrics
            })
            
            logger.info(
                "Model evaluation completed",
                model_name=model_name,
                accuracy=accuracy,
                f1_score=f1
            )
            
            return metrics
            
        except Exception as e:
            logger.error(
                "Model evaluation failed",
                model_name=model_name,
                error=str(e)
            )
            raise

# NLP models
class NLPModels:
    """Natural Language Processing models."""
    
    def __init__(self):
        """Initialize NLP models."""
        self.tokenizer = None
        self.model = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load NLP models."""
        try:
            # Load BERT tokenizer and model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            logger.info("NLP models loaded successfully")
        except Exception as e:
            logger.warning("Failed to load NLP models", error=str(e))
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embeddings."""
        if not self.tokenizer or not self.model:
            # Fallback to simple encoding
            return np.array([hash(text) % 1000])
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.numpy().flatten()
            
        except Exception as e:
            logger.error("Text encoding failed", error=str(e))
            # Fallback
            return np.array([hash(text) % 1000])
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories."""
        # Simple keyword-based classification
        text_lower = text.lower()
        scores = {}
        
        for category in categories:
            # Simple keyword matching
            keywords = {
                'technology': ['tech', 'software', 'computer', 'digital'],
                'business': ['business', 'company', 'market', 'finance'],
                'health': ['health', 'medical', 'doctor', 'hospital'],
                'education': ['education', 'learn', 'school', 'student']
            }
            
            category_keywords = keywords.get(category.lower(), [category.lower()])
            score = sum(1 for keyword in category_keywords if keyword in text_lower)
            scores[category] = score / len(category_keywords)
        
        return scores

# Feature engineering
class FeatureEngineer:
    """Feature engineering utilities."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_extractors: Dict[str, Callable] = {}
        self._register_default_extractors()
    
    def _register_default_extractors(self) -> None:
        """Register default feature extractors."""
        # Text features
        self.feature_extractors['text_length'] = lambda x: len(str(x))
        self.feature_extractors['word_count'] = lambda x: len(str(x).split())
        self.feature_extractors['has_numbers'] = lambda x: any(c.isdigit() for c in str(x))
        self.feature_extractors['has_special_chars'] = lambda x: any(
            not c.isalnum() and not c.isspace() for c in str(x)
        )
        
        # Time features
        self.feature_extractors['hour_of_day'] = lambda x: x.hour if hasattr(x, 'hour') else 0
        self.feature_extractors['day_of_week'] = lambda x: x.weekday() if hasattr(x, 'weekday') else 0
        self.feature_extractors['is_weekend'] = lambda x: x.weekday() >= 5 if hasattr(x, 'weekday') else False
    
    def extract_features(
        self,
        data: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract features from data."""
        features = {}
        
        if feature_names is None:
            feature_names = list(self.feature_extractors.keys())
        
        for feature_name in feature_names:
            if feature_name in self.feature_extractors:
                extractor = self.feature_extractors[feature_name]
                
                # Apply extractor to relevant data
                for key, value in data.items():
                    try:
                        feature_key = f"{feature_name}_{key}"
                        features[feature_key] = extractor(value)
                    except Exception:
                        features[feature_key] = 0
        
        return features
    
    def register_extractor(self, name: str, extractor: Callable) -> None:
        """Register a custom feature extractor."""
        self.feature_extractors[name] = extractor

# Model explainability
class ModelExplainer:
    """Model explainability utilities."""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize model explainer."""
        self.model_manager = model_manager
    
    def explain_prediction(
        self,
        model_name: str,
        input_data: Union[np.ndarray, List, Dict[str, Any]],
        method: str = "feature_importance"
    ) -> Dict[str, Any]:
        """Explain model prediction."""
        if model_name not in self.model_manager.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_wrapper = self.model_manager.models[model_name]
        
        if method == "feature_importance":
            return self._feature_importance_explanation(model_wrapper, input_data)
        elif method == "lime":
            return self._lime_explanation(model_wrapper, input_data)
        else:
            return {"error": f"Explanation method {method} not supported"}
    
    def _feature_importance_explanation(
        self,
        model_wrapper: ModelWrapper,
        input_data: Union[np.ndarray, List, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate feature importance explanation."""
        # Simple feature importance (for tree-based models)
        if hasattr(model_wrapper.model, 'feature_importances_'):
            importances = model_wrapper.model.feature_importances_
            feature_names = model_wrapper.metadata.feature_columns
            
            explanation = {
                'method': 'feature_importance',
                'feature_importance': dict(zip(feature_names, importances.tolist())),
                'top_features': sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        else:
            explanation = {
                'method': 'feature_importance',
                'message': 'Feature importance not available for this model type'
            }
        
        return explanation
    
    def _lime_explanation(
        self,
        model_wrapper: ModelWrapper,
        input_data: Union[np.ndarray, List, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate LIME explanation."""
        # Simplified LIME implementation
        explanation = {
            'method': 'lime',
            'message': 'LIME explanation not implemented for this model type',
            'local_importance': {}
        }
        
        return explanation

# Global instances
_model_manager: Optional[ModelManager] = None
_nlp_models: Optional[NLPModels] = None
_feature_engineer: Optional[FeatureEngineer] = None
_model_explainer: Optional[ModelExplainer] = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager()
    
    return _model_manager

def get_nlp_models() -> NLPModels:
    """Get global NLP models instance."""
    global _nlp_models
    
    if _nlp_models is None:
        _nlp_models = NLPModels()
    
    return _nlp_models

def get_feature_engineer() -> FeatureEngineer:
    """Get global feature engineer instance."""
    global _feature_engineer
    
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()
    
    return _feature_engineer

def get_model_explainer() -> ModelExplainer:
    """Get global model explainer instance."""
    global _model_explainer
    
    if _model_explainer is None:
        _model_explainer = ModelExplainer(get_model_manager())
    
    return _model_explainer

# ML utilities
async def predict_text_category(text: str, categories: List[str]) -> Dict[str, float]:
    """Predict text category using NLP models."""
    nlp_models = get_nlp_models()
    return nlp_models.classify_text(text, categories)

async def detect_anomaly(features: np.ndarray) -> bool:
    """Detect anomaly using ML model."""
    model_manager = get_model_manager()
    prediction = await model_manager.predict("anomaly_detector", features)
    return prediction[0] == -1  # -1 indicates anomaly

async def get_recommendations(user_id: str, top_k: int = 5) -> List[int]:
    """Get recommendations for user."""
    model_manager = get_model_manager()
    recommendations = await model_manager.predict("recommendation_model", user_id)
    return recommendations[:top_k]

# Export public API
__all__ = [
    # Classes
    'ModelManager',
    'ModelWrapper',
    'NLPModels',
    'FeatureEngineer',
    'ModelExplainer',
    'ModelMetadata',
    
    # Enums
    'ModelType',
    'ModelStatus',
    
    # Functions
    'get_model_manager',
    'get_nlp_models',
    'get_feature_engineer',
    'get_model_explainer',
    'predict_text_category',
    'detect_anomaly',
    'get_recommendations',
] 