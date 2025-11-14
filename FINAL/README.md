# Final Status Classification Pipeline

Complete modular pipeline for status classification using ensemble learning with EfficientTCN models and sequence-preserving data augmentation.

## Overview

This is a modular pipeline that can be used in two ways:

1. **Complete Pipeline**: Run full training and evaluation pipeline
2. **Individual Modules**: Use specific modules for inference or other tasks

The pipeline consists of the following modules:

- **augmentation.py**: Generate augmented data using sequence-preserving augmentation
- **training.py**: Train ensemble of models and create train/validation split
- **thresholds.py**: Optimize per-class probability thresholds
- **evaluation.py**: Evaluate ensemble on test set
- **runtime.py**: Measure average inference time
- **inference.py**: Make predictions on single drills (reusable)
- **pipeline.py**: Orchestrates complete pipeline
- **main.py**: Entry point for pipeline or inference

## Complete Pipeline Steps

1. **Data Augmentation**: Generates augmented data using sequence-preserving augmentation
2. **Train/Validation Split**: Creates a validation split ensuring validation samples are not in Option 2 (used for augmentation)
3. **Ensemble Training**: Trains multiple models with different seeds for ensemble learning
4. **Threshold Optimization**: Optimizes per-class probability thresholds on validation set (saves to JSON)
5. **Test Set Evaluation**: Evaluates ensemble on test set with optimized thresholds
6. **Runtime Evaluation**: Measures average inference time in production mode (CPU-only, batch size=1)

## Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

**Note for PyTorch with CUDA**: If you need GPU support, install PyTorch with CUDA first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```
For other CUDA versions, check [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Required packages:
- Python 3.8+
- PyTorch >= 2.0.0 (CPU or CUDA)
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- joblib >= 1.3.0
- tqdm >= 4.65.0
- psutil >= 5.9.0 (optional, for RAM measurements in runtime evaluation)

The pipeline imports modules from:
- `../Status Classification/` - Core classification modules (config, preprocessing, model, train, run_ensemble)
- `../Sequence-Preserving Augmentation/` - Data augmentation module

## Usage

### Complete Pipeline

Run the complete training and evaluation pipeline:
```bash
python main.py --mode pipeline --config config.json
```

### Runtime Evaluation (After Training)

To run only the runtime evaluation on already trained models:
```bash
python main.py --mode runtime --config config.json
```

This will:
- Load the trained ensemble models
- Load the optimized thresholds
- Measure average inference time
- Calculate storage usage (models, scaler, dependencies)
- Measure RAM usage (if psutil is installed)
- Validate hardware requirements (speed < 3s, RAM, storage)

### Single Drill Inference

Make predictions on a single drill CSV file:
```bash
python main.py --mode inference --csv path/to/drill.csv --config config.json
```

You can also specify custom thresholds:
```bash
python main.py --mode inference --csv path/to/drill.csv --thresholds path/to/thresholds.json --config config.json
```

## Configuration

All hyperparameters and settings are defined in `config.json`. Key sections:

### Data Paths
- `train_path`: Path to training data (Option 1)
- `test_path`: Path to test data (Option 1)
- `augmented_data_path`: Path where augmented data is stored/generated
- `option2_train_path`: Path to Option 2 training data (used for augmentation)
- `exclude_files_csv`: CSV file with files to exclude

### Data Augmentation
- `enabled`: Whether to generate augmented data (can be skipped if already exists)
- `target_counts`: Target counts for each class after augmentation

### Preprocessing
- `max_series_length`: Maximum length for padding/truncation (10000)
- `normalize`: Whether to standardize features (true)
- `include_derivatives`: Whether to add derivative features (false)
- `validation_split`: Fraction of data for validation (0.2)

### Model Architecture
- `channels`: Number of channels per layer [64, 128, 256, 512, 512]
- `dilations`: Dilation rates [1, 2, 4, 8, 16]
- `strides`: Stride sizes [2, 2, 2, 2, 1]
- `kernel_size`: Convolution kernel size (7)
- `dropout`: Dropout rate (0.3)
- `activation`: Activation function ('swish')

### Training
- `num_epochs`: Maximum number of training epochs (300)
- `batch_size`: Training batch size (64)
- `learning_rate`: Initial learning rate (0.0005)
- `early_stopping_patience`: Patience for early stopping (25)
- `use_fixed_class_weights`: Use fixed class weights based on original distribution (true)
- `fixed_class_distribution`: Original class distribution for weight calculation

### Ensemble
- `num_models`: Number of models in ensemble (5)
- `model_seeds`: Random seeds for each model [43, 44, 45, 46, 47]

### Threshold Optimization
- `fpr_limits`: Maximum false positive rates per class
- `fnr_limits`: Maximum false negative rates per class
- `min_accuracy`: Minimum accuracy threshold (0.90)
- `threshold_grid_start/end/steps`: Grid search parameters

### Runtime Evaluation
- `hardware_constraints`: CPU-only mode settings
- `num_samples`: Number of test samples for runtime evaluation (100)

## Usage

### Run Complete Pipeline

```bash
cd FINAL
python main.py --mode pipeline
```

Or simply:
```bash
python main.py
```

The pipeline will:
1. Load configuration from `config.json`
2. Generate augmented data (if enabled and not exists)
3. Create train/validation split
4. Train ensemble of 5 models
5. Optimize thresholds on validation set (saves to `results/{experiment_name}_thresholds.json`)
6. Evaluate on test set
7. Measure runtime performance
8. Save all results to `results/{experiment_name}_final_results.json`

### Run Inference on Single Drill

After training, you can use the trained models to make predictions on individual drills:

```bash
python main.py --mode inference --csv path/to/drill.csv
```

This will:
1. Load trained ensemble models
2. Load optimized thresholds
3. Preprocess the drill
4. Make prediction using ensemble
5. Print predicted class and probabilities

### Using Modules Individually

You can also import and use individual modules in your own scripts:

```python
from inference import load_ensemble_models, predict_single_drill
from thresholds import load_thresholds
import json

# Load configuration
with open('config.json', 'r') as f:
    config_dict = json.load(f)

# Load models and preprocessor
models, config, preprocessor = load_ensemble_models(config_dict)

# Load thresholds
class_names = ["Normal", "NPT", "OD"]
thresholds = load_thresholds('results/final_ensemble_thresholds.json', class_names)

# Make prediction
result = predict_single_drill('path/to/drill.csv', models, preprocessor, thresholds, 
                             class_names=class_names)
print(f"Predicted: {result['class_name']}")
```

### Output

The pipeline produces:

- **Model checkpoints**: Saved in `results/{experiment_name}_model_{id}/best_model.pth`
- **Scaler**: Saved in `results/{experiment_name}_scaler.pkl`
- **Thresholds**: Saved in `results/{experiment_name}_thresholds.json`
- **Final results**: Saved in `results/{experiment_name}_final_results.json`

The final results JSON contains:
- Optimized thresholds
- Test set metrics (accuracy, ROC AUC, FPR, FNR per class)
- Runtime statistics (mean/median/std time per drill)
- Individual model performance
- Pipeline execution time

## Results

### Test Set Metrics

- **Accuracy**: Classification accuracy with optimized thresholds
- **ROC AUC**: Macro-averaged one-vs-rest ROC AUC
- **FPR/FNR**: Per-class false positive and false negative rates
- **Constraint Status**: Whether FPR/FNR limits are met

### Runtime Performance

Runtime evaluation measures the complete inference pipeline:
- Preprocessing (normalization, padding/truncation)
- Ensemble prediction (all models, batch size=1)
- Threshold application

Statistics reported:
- Mean/median/std inference time per drill
- Min/max inference time
- Throughput (drills/second)

**Requirement Validation:**
The runtime evaluation validates all speed and hardware requirements:
- **Speed**: Average inference time must be < 3 seconds
- **Storage**: Application storage usage (models + dependencies)
- **RAM**: Peak memory usage during inference (if psutil is installed)

All requirements are checked and reported with PASS/FAIL status.

## Pipeline Steps

### Step 1: Data Augmentation
Generates synthetic time series data using sequence-preserving augmentation from Option 2 data. Target counts defined in configuration. Uses segment-level augmentation to preserve stage sequences.

### Step 2: Train/Validation Split
Ensures validation set contains only samples from Option 1 that are not in Option 2, preventing data leakage from augmentation. Split is shared across all ensemble models.

### Step 3: Ensemble Training
Trains N models (default: 5) with different random seeds. Each model:
- Uses the same train/validation split
- Includes augmented data in training
- Uses original data distribution for class weights (to avoid imbalance from synthetic data)
- Saves best model checkpoint based on validation loss

### Step 4: Threshold Optimization
Searches for optimal per-class probability thresholds on validation set that:
- Meet FPR/FNR constraints
- Achieve minimum accuracy (0.90)
- Maximize accuracy subject to constraints

### Step 5: Test Set Evaluation
Evaluates ensemble on test set using optimized thresholds, reporting:
- Accuracy
- ROC AUC
- Per-class FPR and FNR
- Constraint satisfaction
- Individual model performance

### Step 6: Runtime Evaluation
Measures inference time in production mode (CPU-only, batch size=1) on subset of test files. Evaluates complete pipeline: preprocessing -> ensemble prediction -> threshold application.

## Notes

- All hyperparameters are defined in `config.json` (no hardcoded values)
- Augmentation step can be skipped if data already exists (set `enabled: false` in config)
- Runtime evaluation is optional and can be skipped if it fails
- Models are saved in inference mode (no dropout, gradients disabled)
- Class weights are based on original data distribution, not augmented distribution

