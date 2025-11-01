"""
Main script to run complete status classification experiments
"""
from config import Config
from train import Trainer
from evaluate import Evaluator
import os


def main():
    """Main experiment execution"""
    
    config = Config()
    
    # Create experiment-specific folders
    experiment_dir = os.path.join("results", config.experiment_name)
    config.model_dir = os.path.join(experiment_dir, "models")
    config.results_dir = os.path.join(experiment_dir, "results")
    config.logs_dir = os.path.join(experiment_dir, "logs")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    print("="*80)
    print(f"STARTING EXPERIMENT: {config.experiment_name}")
    print("="*80)
    
    # Print configuration
    print("\n" + "-"*80)
    print("CONFIGURATION")
    print("-"*80)
    print(f"Experiment name:      {config.experiment_name}")
    print(f"Results directory:    {experiment_dir}")
    print()
    print("Preprocessing:")
    print(f"  Max length:         {config.max_series_length}")
    print(f"  Normalize:          {config.normalize}")
    print(f"  Derivatives:        {config.include_derivatives}")
    print()
    print("Architecture:")
    print(f"  Channels:           {config.channels}")
    print(f"  Dilations:          {config.dilations}")
    print(f"  Kernel size:        {config.kernel_size}")
    print(f"  Dropout:            {config.dropout}")
    print(f"  Receptive field:    {config.receptive_field}")
    print()
    print("Training:")
    print(f"  Batch size:         {config.batch_size}")
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Weight decay:       {config.weight_decay}")
    print(f"  Num epochs:         {config.num_epochs}")
    print(f"  Early stopping:     {config.early_stopping_patience}")
    print()
    print("Classes:")
    for name in config.get_class_names():
        print(f"  - {name}")
    
    # Save configuration
    config_path = os.path.join(experiment_dir, f"{config.experiment_name}_config.json")
    config.save_config(config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    # STEP 1: TRAINING
    print("\n" + "="*80)
    print("STEP 1: TRAINING")
    print("="*80)
    
    trainer = Trainer(config)
    trainer.train()
    
    print(f"\nTraining completed. Best validation loss: {trainer.best_val_loss:.4f}")
    
    # STEP 2: EVALUATION
    print("\n" + "="*80)
    print("STEP 2: EVALUATION")
    print("="*80)
    
    evaluator = Evaluator(config, trainer.model, trainer.preprocessor)
    metrics = evaluator.evaluate()
    
    print(f"\nEvaluation completed")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nExperiment folder: {experiment_dir}")
    print(f"\nContents:")
    print(f"  - {config.experiment_name}_config.json")
    print(f"  - models/")
    print(f"  - results/{config.experiment_name}_history.json")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

