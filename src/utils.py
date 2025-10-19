"""
Utility functions for the MVP Diabetes Project
Handles configuration, logging, I/O operations, and validation
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, environment variables won't be loaded from .env")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    print("Warning: loguru not installed, using standard logging")

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed")
    np = None

try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow not installed")
    Image = None

import base64
from io import BytesIO


class Config:
    """Configuration manager for the project"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = "config/settings.yaml"):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Replace environment variables
        self._replace_env_vars(self._config)
        logger.info(f"Configuration loaded from {config_path}")
    
    def _replace_env_vars(self, config: Dict):
        """Recursively replace ${ENV_VAR} with environment variables"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._replace_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, "")
    
    def get(self, path: str, default=None) -> Any:
        """Get configuration value using dot notation (e.g., 'models.segmentation.device')"""
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """Get path from configuration and ensure it exists"""
        path_str = self.get(f"paths.{key}")
        if path_str is None:
            raise ValueError(f"Path '{key}' not found in configuration")
        
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path


class DensityLookup:
    """Lookup table for food densities"""
    
    def __init__(self, densities_path: str = "config/densities.yaml"):
        with open(densities_path, 'r') as f:
            self._densities = yaml.safe_load(f)
        logger.info(f"Loaded densities from {densities_path}")
    
    def get_density(self, food_name: str, category: Optional[str] = None) -> float:
        """
        Get density for a food item
        
        Args:
            food_name: Name of the food (e.g., 'chicken_cooked', 'rice')
            category: Category hint (e.g., 'proteins', 'grains')
        
        Returns:
            Density in g/ml
        """
        food_key = food_name.lower().replace(' ', '_')
        
        # Try exact match in category
        if category:
            category_densities = self._densities.get(category, {})
            if food_key in category_densities:
                return category_densities[food_key]
        
        # Search all categories
        for cat_densities in self._densities.values():
            if isinstance(cat_densities, dict) and food_key in cat_densities:
                return cat_densities[food_key]
        
        # Try partial match
        for cat_densities in self._densities.values():
            if isinstance(cat_densities, dict):
                for key, value in cat_densities.items():
                    if food_key in key or key in food_key:
                        logger.warning(f"Using approximate match for '{food_name}': {key}")
                        return value
        
        # Use default
        default_density = self._densities.get('defaults', {}).get('unknown', 0.8)
        logger.warning(f"No density found for '{food_name}', using default: {default_density}")
        return default_density


def setup_logging(config: Optional[Config] = None):
    """Configure logging for the project"""
    if config is None:
        config = Config()
    
    log_path = config.get_path("logs")
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.format")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_path / "pipeline_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=log_level,
        rotation=config.get("logging.rotation", "100 MB"),
        retention=config.get("logging.retention", "30 days")
    )
    
    logger.info("Logging initialized")


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from path and return as numpy array"""
    img = Image.open(image_path)
    return np.array(img)


def save_image(image: np.ndarray, output_path: Union[str, Path]):
    """Save numpy array as image"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(output_path)
    logger.debug(f"Saved image to {output_path}")


def image_to_base64(image: Union[np.ndarray, Image.Image]) -> str:
    """Convert image to base64 string"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return np.array(img)


def save_json(data: Dict, output_path: Union[str, Path], indent: int = 2):
    """Save dictionary to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)
    
    logger.debug(f"Saved JSON to {output_path}")


def load_json(json_path: Union[str, Path]) -> Dict:
    """Load JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """Validate JSON data against schema"""
    from jsonschema import validate, ValidationError
    
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        logger.error(f"JSON validation error: {e}")
        return False


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(predictions - targets))


class PipelineTimer:
    """Context manager for timing pipeline operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Completed: {self.operation_name} ({duration:.2f}s)")
        return False
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# JSON Schemas for validation
SEGMENTATION_OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "mask_id": {"type": "integer"},
            "crop_image_path": {"type": "string"},
            "volume_ml": {"type": "number"}
        },
        "required": ["mask_id", "crop_image_path", "volume_ml"]
    }
}

CLASSIFICATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "mask_id": {"type": "integer"},
        "dish_name": {"type": "string"},
        "main_ingredients": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["mask_id", "dish_name", "main_ingredients", "confidence"]
}

NUTRITION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_food": {"type": "string"},
        "carbohydrates_per_100g": {"type": "number"},
        "protein_per_100g": {"type": "number"},
        "fat_per_100g": {"type": "number"},
        "match_score": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["matched_food", "carbohydrates_per_100g", "protein_per_100g", "fat_per_100g", "match_score"]
}

FINAL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "input_image": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mask_id": {"type": "integer"},
                    "dish_name": {"type": "string"},
                    "volume_ml": {"type": "number"},
                    "nutrition": {"type": "object"},
                    "confidence": {"type": "number"}
                }
            }
        },
        "total": {
            "type": "object",
            "properties": {
                "carbohydrates_g": {"type": "number"},
                "protein_g": {"type": "number"},
                "fat_g": {"type": "number"}
            }
        },
        "overall_confidence": {"type": "number"}
    },
    "required": ["input_image", "segments", "total", "overall_confidence"]
}