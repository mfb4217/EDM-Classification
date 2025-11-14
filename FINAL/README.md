# EDM Drill Status Detection

Automatic drilling status classification system using deep learning. The system analyzes time series of drilling data (Voltage and Depth) and classifies each drill into one of three categories: **Normal**, **NPT**, or **OD**.

## Approach

Our solution combines **domain knowledge-based data augmentation** with a **Temporal Convolutional Network (TCN)** architecture:

### 1. Domain Knowledge-Based Data Augmentation

The system uses sequence-preserving augmentation that respects the physical characteristics of drilling operations. Instead of random transformations, augmentation:
- Preserves the exact sequence of drilling stages (each stage has specific characteristics)
- Maintains temporal relationships between Voltage and Depth measurements
- Generates synthetic examples by recombining valid stage segments from Option 2 training data
- Ensures synthetic data follows realistic drilling patterns observed in real operations

This approach balances classes while maintaining the integrity of temporal patterns, which is crucial for accurate classification.

### 2. Temporal Convolutional Network (TCN)

The model processes time series of **Voltage** and **Depth** measurements using a deep convolutional architecture:

- **Input**: Time series of Voltage and Depth (2 channels, up to 10,000 time steps)
- **Dilated Convolutions**: Capture patterns at multiple temporal scales (dilations: 1, 2, 4, 8, 16)
  - Each layer increases the receptive field exponentially while maintaining computational efficiency
- **Stride Convolutions**: Progressively reduce sequence length (strides: 2, 2, 2, 2, 1)
  - Reduces computation and increases abstraction level through the network
- **Residual Connections**: Enable stable training of deep networks
  - Skip connections help gradients flow and prevent degradation in deeper layers
- **Convolutional to Classification**: 
  - Features extracted through convolutional layers are aggregated via global pooling
  - Dense layers perform final classification into Normal, NPT, or OD classes

The ensemble approach (5 models) averages predictions for improved robustness and generalization.

## Results

### Test Set Performance

The trained ensemble achieves the following performance on the test set:

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.93% |
| **ROC AUC** (macro-average) | 97.68% |

#### Per-Class Performance

| Class | FPR (False Positive Rate) | FNR (False Negative Rate) |
|-------|---------------------------|---------------------------|
| **Normal** | 8.76% | 6.67% |
| **NPT** | 3.56% | 18.18% |
| **OD** | 2.17% | 2.82% |

### Runtime Performance

Inference performance measured on the following hardware configuration:
- **CPU**: 4 cores
- **GPU**: None (CPU-only)
- **RAM**: 8 GB (peak usage: 1.62 GB)
- **Storage**: 120 GB SSD (application usage: 3.08 GB)

| Metric | Value |
|--------|-------|
| **Average Runtime** | 82 ms (0.082 seconds) per drill |
| **Median Runtime** | 81 ms |
| **Throughput** | 12.2 drills/second |
| **Speed Requirement** | < 3 seconds (✓ Passed) |

**All hardware requirements met**: Speed, RAM, and Storage constraints are satisfied.

## Requirements

### Quick Installation

```bash
# Install basic dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

If you have an NVIDIA GPU and want to speed up training:

```bash
# First install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then install the rest
pip install -r requirements.txt
```

**Note**: Training works with CPU but is slower. Production inference is optimized for CPU.

## Quick Usage

### 1. Train Complete Model

```bash
cd FINAL
python main.py
```

This runs the complete pipeline:
- Generates augmented data (if not exists)
- Trains ensemble of 5 models
- Optimizes classification thresholds on the validation set
- Evaluates on test set
- Measures inference performance

### 2. Classify a Single Drill

```bash
python main.py --mode inference --csv path/to/drill.csv
```

Example output:
```
Predicted class: NPT
Probabilities:
  Normal: 0.12
  NPT: 0.85
  OD: 0.03
```

### 3. Evaluate Performance Only (After Training)

```bash
python main.py --mode runtime
```

Measures inference speed, RAM usage, and storage.

## Configuration

Everything is configured in `config.json`.

### Data Paths

```json
{
  "data_paths": {
    "train_path": "../Data/Option 1/Train",
    "test_path": "../Data/Option 1/Test",
    "augmented_data_path": "../Augmented Data",
    "option2_train_path": "../Data/Option 2/Train",
    "exclude_files_csv": "../Data/files_to_remove_due_to_double_drill_error.csv"
  }
}
```

See complete `config.json` for all parameters.

## Project Structure

```
GE/
├── FINAL/                          # Main project directory
│   ├── main.py                     # Main entry point
│   ├── pipeline.py                 # Complete pipeline orchestration
│   ├── config.json                 # Configuration (hyperparameters, paths)
│   │
│   ├── augmentation.py             # Synthetic data generation
│   ├── preprocessing.py            # Data preprocessing
│   ├── train.py                    # Individual model training
│   ├── training.py                 # Ensemble training
│   ├── model.py                    # EfficientTCN architecture
│   │
│   ├── thresholds.py               # Threshold optimization
│   ├── evaluation.py               # Test set evaluation
│   ├── runtime.py                  # Performance measurement
│   ├── inference.py                # Production prediction
│   │
│   ├── config.py                   # Configuration utilities
│   ├── utils.py                    # Helper functions
│   ├── requirements.txt            # Python dependencies
│   │
│   └── results/                    # Results (models, metrics, thresholds)
│       ├── final_ensemble_model_01/
│       │   └── best_model.pth
│       ├── final_ensemble_scaler.pkl
│       ├── final_ensemble_thresholds.json
│       └── final_ensemble_final_results.json
│
├── Data/
│   ├── Option 1/
│   │   ├── Train/
│   │   │   ├── Normal/
│   │   │   ├── NPT/
│   │   │   └── OD/
│   │   └── Test/
│   │       ├── Normal/
│   │       ├── NPT/
│   │       └── OD/
│   │
│   ├── Option 2/
│   │   └── Train/
│   │       ├── Normal/
│   │       ├── NPT/
│   │       └── OD/
│   │
│   └── files_to_remove_due_to_double_drill_error.csv  # Exclusion list
│
└── Augmented Data/                 # Generated by augmentation step
```

**CSV File Format**: Each CSV file should contain columns `Voltage` and `Z` (Depth) with time series data.

## Detailed training pipeline Explanation

### Step 1: Data Augmentation
Generates synthetic data preserving drilling stage sequences. Uses Option 2 training data as source and maintains real temporal structure. This helps with generalization and balancing classes without introducing unrealistic patterns.

### Step 2: Train/Validation Split
Splits data into training (80%) and validation (20%), ensuring:
- Validation data is NOT in Option 2 (used for augmentation)
- Split is consistent for all ensemble models

### Step 3: Ensemble Training
Trains multiple models (default: 5) with different random seeds. Each model:
- Uses the same train/validation split
- Trains with original + augmented data
- Class weights calculated on original distribution (not augmented)
- Only best model epoch saved based on validation

**Why ensemble?** Combines multiple models for greater robustness and better generalization.

### Step 4: Threshold Optimization
Searches for per-class probability thresholds that meet:
- Maximum FPR (False Positive Rate) per class
- Maximum FNR (False Negative Rate) per class
- Minimum accuracy of 90%

Thresholds are optimized on validation set and then applied to test.


### Step 5: Test Set Evaluation
Evaluates ensemble on test set with optimized thresholds. Reports:
- Global accuracy
- ROC AUC (macro-average)
- FPR and FNR per class
- Confusion matrix

### Step 6: Performance Evaluation
Measures inference time in production mode (CPU, batch=1) on test subset. Validates:
- **Speed**: Average < 3 seconds per drill
- **RAM**: Maximum usage during inference
- **Storage**: Model size + dependencies

### Output Files

After training, results are saved in `results/`:

- **`final_ensemble_final_results.json`**: Complete metrics, optimized thresholds, performance
- **`final_ensemble_thresholds.json`**: Optimized thresholds (for production use)
- **`final_ensemble_scaler.pkl`**: Pre-trained scaler (for consistent normalization)
- **`final_ensemble_model_XX/best_model.pth`**: Weights of each ensemble model

## Technical Notes

### Data Leakage Prevention
- Scaler fitted only on train and saved for inference
- Validation split excludes Option 2 (used for augmentation)
- Test set not used until final evaluation (just inference)

### Model Architecture
EfficientTCN (Temporal Convolutional Network) processes Voltage and Depth time series through:
- **Dilated convolutions**: Exponential receptive field expansion (dilations: 1, 2, 4, 8, 16)
- **Stride convolutions**: Progressive length reduction (strides: 2, 2, 2, 2, 1) for efficiency
- **Residual connections**: Stable deep network training with skip connections
- **Global pooling + dense layers**: Feature aggregation and final classification

## Troubleshooting

**Error: "No model checkpoints found"**
- Make sure you trained first: `python main.py --mode pipeline`

**Error: "Preprocessor file not found"**
- Scaler is saved during training. Re-train if missing.

**Inference very slow**
- Check you're in CPU mode (as in production)
- Reduce `num_samples` in `runtime_evaluation` for testing


## Support

For issues or questions, check:
- `config.json` to adjust parameters
- Training logs for errors
- `final_results.json` for detailed metrics
