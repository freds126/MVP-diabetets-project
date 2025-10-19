"""
LangGraph Pipeline Orchestration
Manages the end-to-end pipeline execution
"""

from typing import Dict, Optional, TypedDict, Any
from pathlib import Path
from loguru import logger

from .utils import Config, setup_logging, save_json, get_timestamp
from .segmenter import FoodSegmenter
from .classifier import FoodClassifier
from .nutrition_search import NutritionSearch
from .aggregator import NutrientAggregator


class PipelineState(TypedDict):
    """State object passed between pipeline nodes"""
    image_path: str
    output_dir: str
    segmentation_results: Optional[list]
    classification_results: Optional[list]
    nutrition_results: Optional[list]
    final_results: Optional[dict]
    errors: list
    metadata: dict


class NutritionPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        setup_logging(self.config)
        
        # Initialize components
        self.segmenter = FoodSegmenter(self.config)
        self.classifier = FoodClassifier(self.config)
        self.nutrition_search = NutritionSearch(self.config)
        self.aggregator = NutrientAggregator(self.config)
        
        logger.info("NutritionPipeline initialized")
    
    def run(
        self,
        image_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            image_path: Path to input meal image
            output_dir: Directory to save outputs
        
        Returns:
            Final aggregated results
        """
        # Setup output directory
        if output_dir is None:
            timestamp = get_timestamp()
            output_dir = self.config.get_path("outputs") / f"run_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info(f"Starting Nutrition Pipeline")
        logger.info(f"Input Image: {image_path}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info("=" * 80)
        
        # Initialize state
        state: PipelineState = {
            "image_path": image_path,
            "output_dir": str(output_dir),
            "segmentation_results": None,
            "classification_results": None,
            "nutrition_results": None,
            "final_results": None,
            "errors": [],
            "metadata": {
                "pipeline_version": "1.0",
                "timestamp": get_timestamp()
            }
        }
        
        # Execute pipeline nodes
        try:
            state = self._node_segment(state)
            state = self._node_classify(state)
            state = self._node_nutrition_search(state)
            state = self._node_aggregate(state)
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            state["errors"].append(str(e))
            raise
        
        # Save complete state
        state_path = output_dir / "pipeline_state.json"
        save_json(state, state_path)
        logger.info(f"Saved pipeline state to {state_path}")
        
        return state["final_results"]
    
    def _node_segment(self, state: PipelineState) -> PipelineState:
        """Segmentation node"""
        logger.info("\n[NODE] Segmentation + Volume Estimation")
        
        try:
            results = self.segmenter.process(
                image_path=state["image_path"],
                output_dir=Path(state["output_dir"]) / "segmentation"
            )
            state["segmentation_results"] = results
            logger.info(f"✓ Segmentation complete: {len(results)} segments found")
            
        except Exception as e:
            logger.error(f"✗ Segmentation failed: {e}")
            state["errors"].append(f"Segmentation: {e}")
            raise
        
        return state
    
    def _node_classify(self, state: PipelineState) -> PipelineState:
        """Classification node"""
        logger.info("\n[NODE] Food Classification")
        
        if not state["segmentation_results"]:
            logger.warning("No segmentation results, skipping classification")
            return state
        
        try:
            results = self.classifier.process(
                segmentation_results=state["segmentation_results"],
                output_dir=Path(state["output_dir"]) / "classification"
            )
            state["classification_results"] = results
            logger.info(f"✓ Classification complete: {len(results)} items classified")
            
        except Exception as e:
            logger.error(f"✗ Classification failed: {e}")
            state["errors"].append(f"Classification: {e}")
            raise
        
        return state
    
    def _node_nutrition_search(self, state: PipelineState) -> PipelineState:
        """Nutrition search node"""
        logger.info("\n[NODE] Nutrition Database Search")
        
        if not state["classification_results"]:
            logger.warning("No classification results, skipping nutrition search")
            return state
        
        try:
            results = self.nutrition_search.process(
                classification_results=state["classification_results"],
                output_dir=Path(state["output_dir"]) / "nutrition_search"
            )
            state["nutrition_results"] = results
            logger.info(f"✓ Nutrition search complete: {len(results)} items matched")
            
        except Exception as e:
            logger.error(f"✗ Nutrition search failed: {e}")
            state["errors"].append(f"Nutrition search: {e}")
            raise
        
        return state
    
    def _node_aggregate(self, state: PipelineState) -> PipelineState:
        """Aggregation node"""
        logger.info("\n[NODE] Nutrient Aggregation")
        
        if not state["nutrition_results"]:
            logger.warning("No nutrition results, skipping aggregation")
            return state
        
        try:
            results = self.aggregator.process(
                nutrition_results=state["nutrition_results"],
                image_path=state["image_path"],
                output_dir=Path(state["output_dir"]) / "aggregation"
            )
            state["final_results"] = results
            logger.info("✓ Aggregation complete")
            
            # Calculate additional metrics
            ratios = self.aggregator.calculate_macronutrient_ratios(results)
            glycemic = self.aggregator.get_glycemic_estimate(results)
            
            state["final_results"]["macronutrient_ratios"] = ratios
            state["final_results"]["glycemic_estimate"] = glycemic
            
            # Export to CSV
            csv_path = Path(state["output_dir"]) / "results.csv"
            self.aggregator.export_to_csv(results, csv_path)
            
        except Exception as e:
            logger.error(f"✗ Aggregation failed: {e}")
            state["errors"].append(f"Aggregation: {e}")
            raise
        
        return state
    
    def run_node(
        self,
        node_name: str,
        state: PipelineState
    ) -> PipelineState:
        """
        Run a specific node independently
        
        Args:
            node_name: Name of node to run ('segment', 'classify', 'nutrition_search', 'aggregate')
            state: Current pipeline state
        
        Returns:
            Updated state
        """
        node_map = {
            "segment": self._node_segment,
            "classify": self._node_classify,
            "nutrition_search": self._node_nutrition_search,
            "aggregate": self._node_aggregate
        }
        
        if node_name not in node_map:
            raise ValueError(f"Unknown node: {node_name}")
        
        logger.info(f"Running node: {node_name}")
        return node_map[node_name](state)
    
    def get_pipeline_summary(self, state: PipelineState) -> Dict:
        """
        Get summary of pipeline execution
        
        Args:
            state: Pipeline state
        
        Returns:
            Summary dictionary
        """
        summary = {
            "status": "success" if not state["errors"] else "failed",
            "errors": state["errors"],
            "segments_found": len(state["segmentation_results"]) if state["segmentation_results"] else 0,
            "items_classified": len(state["classification_results"]) if state["classification_results"] else 0,
            "nutrition_matches": len(state["nutrition_results"]) if state["nutrition_results"] else 0,
            "has_final_results": state["final_results"] is not None
        }
        
        if state["final_results"]:
            summary.update({
                "total_carbs_g": state["final_results"]["total"]["carbohydrates_g"],
                "total_protein_g": state["final_results"]["total"]["protein_g"],
                "total_fat_g": state["final_results"]["total"]["fat_g"],
                "total_energy_kcal": state["final_results"]["total"]["energy_kcal"],
                "overall_confidence": state["final_results"]["overall_confidence"]
            })
        
        return summary


class SimplePipeline:
    """Simplified pipeline without LangGraph for basic usage"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize simple pipeline"""
        self.pipeline = NutritionPipeline(config)
    
    def analyze_meal(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze a meal image and return nutrition info
        
        Args:
            image_path: Path to meal image
            output_dir: Optional output directory
        
        Returns:
            Nutrition analysis results
        """
        return self.pipeline.run(image_path, output_dir)


def create_pipeline(config: Optional[Config] = None) -> NutritionPipeline:
    """
    Factory function to create a pipeline instance
    
    Args:
        config: Optional configuration object
    
    Returns:
        Initialized pipeline
    """
    return NutritionPipeline(config)
