#!/usr/bin/env python3
"""
Main CLI interface for the Nutrition Pipeline
"""

import argparse
import sys
from pathlib import Path

from src import create_pipeline, Config, setup_logging
from src.utils import load_json


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI Pipeline for Meal Recognition and Nutritional Estimation"
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the meal image"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--node",
        type=str,
        choices=["segment", "classify", "nutrition_search", "aggregate"],
        default=None,
        help="Run only a specific pipeline node"
    )
    
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Load previous pipeline state (for resuming or running specific nodes)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Load configuration
    config = Config()
    if args.config != "config/settings.yaml":
        config.load_config(args.config)
    
    # Setup logging
    if args.verbose:
        config._config["logging"]["level"] = "DEBUG"
    setup_logging(config)
    
    # Create pipeline
    pipeline = create_pipeline(config)
    
    try:
        if args.state:
            # Load previous state
            state = load_json(args.state)
            
            if args.node:
                # Run specific node
                print(f"\nRunning node: {args.node}")
                state = pipeline.run_node(args.node, state)
                
                # Save updated state
                from src.utils import save_json
                save_json(state, args.state)
                print(f"Updated state saved to: {args.state}")
            else:
                print("Error: --node must be specified when using --state")
                sys.exit(1)
        
        else:
            # Run complete pipeline
            print("\n" + "="*80)
            print("AI Nutrition Pipeline")
            print("="*80)
            print(f"Input: {args.image_path}")
            
            results = pipeline.run(
                image_path=args.image_path,
                output_dir=args.output
            )
            
            # Display summary
            print("\n" + "="*80)
            print("RESULTS SUMMARY")
            print("="*80)
            print(f"Total Segments: {results['num_segments']}")
            print(f"Total Mass: {results['total']['total_mass_g']:.1f}g")
            print(f"\nMacronutrients:")
            print(f"  Carbohydrates: {results['total']['carbohydrates_g']:.1f}g")
            print(f"  Protein: {results['total']['protein_g']:.1f}g")
            print(f"  Fat: {results['total']['fat_g']:.1f}g")
            print(f"  Energy: {results['total']['energy_kcal']:.1f} kcal")
            print(f"\nConfidence: {results['overall_confidence']:.1%}")
            
            if 'macronutrient_ratios' in results:
                ratios = results['macronutrient_ratios']
                print(f"\nMacro Ratios (% of calories):")
                print(f"  Carbs: {ratios['carbohydrates_percent']:.1f}%")
                print(f"  Protein: {ratios['protein_percent']:.1f}%")
                print(f"  Fat: {ratios['fat_percent']:.1f}%")
            
            print("\n" + "="*80)
            print(f"\nResults saved to: {args.output or 'outputs/'}")
            print("="*80)
    
    except Exception as e:
        print(f"\nError: Pipeline failed - {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
