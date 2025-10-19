"""
MVP Diabetes Project - AI Pipeline for Meal Recognition and Nutritional Estimation
"""

__version__ = "1.0.0"
__author__ = "Fredrick Carlsåker"

from .utils import Config, setup_logging
from .segmenter import FoodSegmenter
from .classifier import FoodClassifier
from .nutrition_search import NutritionSearch
from .aggregator import NutrientAggregator
from .graph import NutritionPipeline, SimplePipeline, create_pipeline

__all__ = [
    "Config",
    "setup_logging",
    "FoodSegmenter",
    "FoodClassifier",
    "NutritionSearch",
    "NutrientAggregator",
    "NutritionPipeline",
    "SimplePipeline",
    "create_pipeline",
]
