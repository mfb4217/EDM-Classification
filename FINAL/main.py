"""
Main Entry Point
Run complete pipeline or individual modules
"""
import argparse
import json
import os
import sys

from pipeline import run_complete_pipeline
from inference import load_ensemble_models, predict_single_drill
from runtime import evaluate_runtime
from thresholds import load_thresholds


def main():
    parser = argparse.ArgumentParser(description='Status Classification Pipeline')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, choices=['pipeline', 'inference', 'runtime'],
                       default='pipeline',
                       help='Mode: pipeline (full training), inference (single prediction), or runtime (runtime evaluation)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file for inference mode')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Path to thresholds JSON file (default: auto-detect from results)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    if args.mode == 'pipeline':
        # Run complete pipeline
        run_complete_pipeline(config_path)
    
    elif args.mode == 'inference':
        # Run inference on single drill
        if args.csv is None:
            print("Error: --csv argument required for inference mode")
            sys.exit(1)
        
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)
        
        # Find thresholds file
        if args.thresholds is None:
            results_dir = config_dict['output_paths']['results_dir']
            experiment_name = config_dict['experiment_name']
            thresholds_path = os.path.join(results_dir, f"{experiment_name}_thresholds.json")
            if not os.path.exists(thresholds_path):
                print(f"Error: Thresholds file not found: {thresholds_path}")
                print("Please run pipeline first or specify --thresholds")
                sys.exit(1)
        else:
            thresholds_path = args.thresholds
        
        print("="*80)
        print("INFERENCE MODE")
        print("="*80)
        print(f"CSV file: {args.csv}")
        print(f"Thresholds: {thresholds_path}")
        
        # Load models
        print("\nLoading ensemble models...")
        models, config, preprocessor = load_ensemble_models(config_dict)
        
        # Load thresholds
        class_names = ["Normal", "NPT", "OD"]
        from thresholds import load_thresholds
        thresholds = load_thresholds(thresholds_path, class_names)
        
        # Make prediction
        print("\nMaking prediction...")
        result = predict_single_drill(args.csv, models, preprocessor, thresholds, 
                                     class_names=class_names)
        
        print("\n" + "="*80)
        print("PREDICTION RESULT")
        print("="*80)
        print(f"Predicted class: {result['class_name']}")
        print(f"\nProbabilities:")
        for class_name, prob in result['probabilities_dict'].items():
            print(f"  {class_name}: {prob:.4f}")
    
    elif args.mode == 'runtime':
        # Run runtime evaluation on trained models
        results_dir = config_dict['output_paths']['results_dir']
        experiment_name = config_dict['experiment_name']
        
        # Find thresholds file
        if args.thresholds is None:
            thresholds_path = os.path.join(results_dir, f"{experiment_name}_thresholds.json")
        else:
            thresholds_path = args.thresholds
        
        if not os.path.exists(thresholds_path):
            print(f"Error: Thresholds file not found: {thresholds_path}")
            print("Please run pipeline first or specify --thresholds")
            sys.exit(1)
        
        print("="*80)
        print("RUNTIME EVALUATION")
        print("="*80)
        print(f"Experiment: {experiment_name}")
        print(f"Thresholds: {thresholds_path}")
        
        # Load models
        print("\nLoading ensemble models...")
        models, config, preprocessor = load_ensemble_models(config_dict)
        
        # Load thresholds
        class_names = ["Normal", "NPT", "OD"]
        thresholds = load_thresholds(thresholds_path, class_names)
        
        # Create minimal all_results structure (only need models for runtime evaluation)
        import torch
        all_results = []
        for model in models:
            # Create minimal result dict with just the model
            all_results.append({'model': model})
        
        # Run runtime evaluation
        print("\nRunning runtime evaluation...")
        runtime_metrics = evaluate_runtime(config_dict, all_results, thresholds)
        
        print("\n" + "="*80)
        print("RUNTIME EVALUATION COMPLETE")
        print("="*80)
        print(f"\nResults saved in configuration and final results.")


if __name__ == "__main__":
    main()

