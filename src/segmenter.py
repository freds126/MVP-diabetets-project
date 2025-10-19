"""
Segmentation and Volume Estimation Module
Handles food segmentation and volume estimation from images
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from loguru import logger

from .utils import Config, save_json, PipelineTimer, SEGMENTATION_OUTPUT_SCHEMA, validate_json_schema
from .models.segmentation_model import SegmentationModel
from .models.depth_model import DepthEstimationModel


class FoodSegmenter:
    """Food segmentation and volume estimation pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize segmenter
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.segmentation_model = SegmentationModel(self.config)
        self.depth_model = DepthEstimationModel(self.config)
        
        self.volume_method = self.config.get("volume_estimation.method", "depth_based")
        self.plate_diameter_cm = self.config.get("volume_estimation.plate_diameter_cm", 26.0)
        self.depth_scale_factor = self.config.get("volume_estimation.depth_scale_factor", 1.0)
        self.min_volume_ml = self.config.get("volume_estimation.min_volume_ml", 5.0)
        self.max_volume_ml = self.config.get("volume_estimation.max_volume_ml", 1000.0)
        
        logger.info(f"FoodSegmenter initialized with {self.volume_method} volume estimation")
    
    def process(
        self, 
        image_path: str,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Process image and return segmentation with volume estimates
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save intermediate outputs
        
        Returns:
            List of segment dictionaries with format:
            [
                {
                    "mask_id": int,
                    "crop_image_path": str,
                    "volume_ml": float,
                    "mask_area_pixels": int,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float
                }
            ]
        """
        with PipelineTimer("Segmentation and Volume Estimation"):
            # Load image
            from .utils import load_image
            image = load_image(image_path)
            logger.info(f"Processing image: {image_path}, shape: {image.shape}")
            
            # Setup output directory
            if output_dir is None:
                output_dir = self.config.get_path("processed") / Path(image_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Segment image
            segments = self.segmentation_model.segment(image)
            logger.info(f"Found {len(segments)} segments")
            
            # Step 2: Extract crops
            crops = self.segmentation_model.extract_crops(
                image, 
                segments,
                output_dir / "crops"
            )
            
            # Step 3: Estimate depth (if using depth-based method)
            depth_map = None
            if self.volume_method == "depth_based":
                depth_map = self.depth_model.estimate_depth(image)
                
                # Save depth visualization
                self.depth_model.visualize_depth(
                    depth_map,
                    output_path=output_dir / "depth_map.png"
                )
            
            # Step 4: Estimate volumes
            results = []
            for i, (segment, (crop, crop_path)) in enumerate(zip(segments, crops)):
                mask = segment["mask"]
                
                # Estimate volume
                volume_ml = self.estimate_volume(
                    mask=mask,
                    depth_map=depth_map,
                    image_shape=image.shape
                )
                
                result = {
                    "mask_id": i,
                    "crop_image_path": str(crop_path) if crop_path else "",
                    "volume_ml": float(volume_ml),
                    "mask_area_pixels": int(mask.sum()),
                    "bbox": segment["bbox"].tolist(),
                    "confidence": float(segment["score"])
                }
                
                results.append(result)
            
            # Validate output schema
            if not validate_json_schema(results, SEGMENTATION_OUTPUT_SCHEMA):
                logger.warning("Output does not match expected schema")
            
            # Save results
            output_json = output_dir / "segmentation_output.json"
            save_json(results, output_json)
            logger.info(f"Saved segmentation results to {output_json}")
            
            return results
    
    def estimate_volume(
        self,
        mask: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple] = None
    ) -> float:
        """
        Estimate volume of a segment
        
        Args:
            mask: Binary mask of the segment
            depth_map: Optional depth map for depth-based estimation
            image_shape: Shape of original image
        
        Returns:
            Estimated volume in ml
        """
        if self.volume_method == "depth_based" and depth_map is not None:
            volume = self._estimate_volume_depth_based(mask, depth_map)
        elif self.volume_method == "plate_reference":
            volume = self._estimate_volume_plate_reference(mask, image_shape)
        else:
            # Fallback to simple area-based estimation
            volume = self._estimate_volume_area_based(mask)
        
        # Clamp to valid range
        volume = max(self.min_volume_ml, min(volume, self.max_volume_ml))
        
        return volume
    
    def _estimate_volume_depth_based(
        self, 
        mask: np.ndarray, 
        depth_map: np.ndarray
    ) -> float:
        """
        Estimate volume using depth information
        
        Args:
            mask: Binary mask
            depth_map: Depth map
        
        Returns:
            Volume in ml
        """
        # Get depth values within mask
        depth_values = depth_map[mask > 0]
        
        if len(depth_values) == 0:
            return self.min_volume_ml
        
        # Calculate volume as sum of depth * pixel_area
        # This is a simplified approach - real implementation would need calibration
        pixel_area_cm2 = 0.1  # Assumed pixel size (needs calibration)
        mean_depth_cm = np.mean(depth_values) * self.depth_scale_factor
        
        volume_cm3 = mask.sum() * pixel_area_cm2 * mean_depth_cm
        volume_ml = volume_cm3  # 1 cm³ = 1 ml
        
        return float(volume_ml)
    
    def _estimate_volume_plate_reference(
        self, 
        mask: np.ndarray,
        image_shape: Tuple
    ) -> float:
        """
        Estimate volume using plate as reference
        
        Args:
            mask: Binary mask
            image_shape: Shape of image (H, W, C)
        
        Returns:
            Volume in ml
        """
        # Estimate pixel-to-cm ratio from plate diameter
        # Assume plate occupies ~70% of image width
        image_width_pixels = image_shape[1]
        plate_diameter_pixels = image_width_pixels * 0.7
        pixels_per_cm = plate_diameter_pixels / self.plate_diameter_cm
        
        # Calculate area in cm²
        area_pixels = mask.sum()
        area_cm2 = area_pixels / (pixels_per_cm ** 2)
        
        # Assume average height of 2 cm for typical food items
        assumed_height_cm = 2.0
        volume_cm3 = area_cm2 * assumed_height_cm
        volume_ml = volume_cm3
        
        return float(volume_ml)
    
    def _estimate_volume_area_based(self, mask: np.ndarray) -> float:
        """
        Simple area-based volume estimation (fallback)
        
        Args:
            mask: Binary mask
        
        Returns:
            Volume in ml
        """
        # Very rough estimation: assume 1000 pixels ≈ 10 ml
        area_pixels = mask.sum()
        volume_ml = (area_pixels / 1000.0) * 10.0
        
        return float(volume_ml)
    
    def visualize_segmentation(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of segmentation results
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
        
        Returns:
            Visualization image
        """
        from .utils import load_image
        
        image = load_image(image_path)
        segments = self.segmentation_model.segment(image)
        
        vis_image = self.segmentation_model.visualize_segments(
            image, 
            segments,
            output_path
        )
        
        return vis_image
