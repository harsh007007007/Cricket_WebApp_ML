"""
Machine Learning module for Cricket Analytics
Exposes key classes and functions for easy import
"""

# Just define what should be exposed from this package
__all__ = [
    'CricketPerformanceModel',
    'train_models',
    'generate_all_visualizations'
]

# Import after __all__ to avoid circular references
from app.ML.ml_models import CricketPerformanceModel
from app.ML.train_models import main as train_models
from app.ML.visualization_generator import generate_all_visualizations
