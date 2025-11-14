"""
Runtime Evaluation Module
Measure average inference time in production mode
Validates speed and hardware requirements
Uses inference module to avoid code duplication
"""
import sys
import os
import numpy as np
import torch
import glob
import time
from tqdm import tqdm

# Try to import psutil for memory measurement
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory measurements will be limited.")

from inference import load_ensemble_models, predict_single_drill


def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total


def calculate_storage_usage(config_dict, results_dir, experiment_name):
    """Calculate storage used by models, scaler, and estimate dependencies"""
    storage_info = {}
    
    # Calculate model storage
    model_size = 0
    num_models = config_dict['ensemble']['num_models']
    for i in range(1, num_models + 1):
        # Models are saved directly in model directory (no nested models/ folder)
        model_path = os.path.join(results_dir, f"{experiment_name}_model_{i:02d}", 
                                 "best_model.pth")
        if os.path.exists(model_path):
            model_size += os.path.getsize(model_path)
    
    storage_info['models_mb'] = model_size / (1024 * 1024)
    
    # Calculate scaler size
    scaler_path = os.path.join(results_dir, f"{experiment_name}_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler_size = os.path.getsize(scaler_path)
        storage_info['scaler_mb'] = scaler_size / (1024 * 1024)
    else:
        storage_info['scaler_mb'] = 0
    
    # Estimate Python + PyTorch + dependencies (rough estimate: 2-5GB typically)
    # This is an approximation - actual size varies by installation
    storage_info['estimated_dependencies_gb'] = 3.0  # Conservative estimate
    
    # Calculate thresholds file
    thresholds_path = os.path.join(results_dir, f"{experiment_name}_thresholds.json")
    if os.path.exists(thresholds_path):
        storage_info['thresholds_kb'] = os.path.getsize(thresholds_path) / 1024
    else:
        storage_info['thresholds_kb'] = 0
    
    # Total application storage (models + scaler + dependencies estimate)
    storage_info['total_application_gb'] = (
        storage_info['models_mb'] / 1024 + 
        storage_info['scaler_mb'] / 1024 + 
        storage_info['estimated_dependencies_gb']
    )
    
    return storage_info


def measure_memory_usage():
    """Measure current memory usage in GB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 ** 3)  # Convert to GB
    return None


def evaluate_runtime(config_dict, all_results, thresholds):
    """
    Evaluate average runtime in production mode (CPU-only, batch size=1).
    Validates speed requirement (< 3 seconds) and hardware constraints.
    
    Args:
        config_dict: Configuration dictionary
        all_results: List of results from each model (contains trained models)
        thresholds: Array of optimized thresholds
        
    Returns:
        Dictionary with runtime statistics and requirement validation
    """
    print("\n" + "="*80)
    print("RUNTIME EVALUATION")
    print("="*80)
    
    runtime_config = config_dict.get('runtime_evaluation', {})
    hardware = runtime_config.get('hardware_constraints', {})
    
    # Requirements
    required_ram_gb = hardware.get('ram_gb', 8)
    required_storage_gb = hardware.get('storage_gb', 120)
    required_max_time_seconds = 3.0
    
    print(f"\nHardware Requirements:")
    print(f"  RAM: {required_ram_gb}GB (system total)")
    print(f"  Storage: {required_storage_gb}GB SSD (system total)")
    print(f"  Speed: < {required_max_time_seconds} seconds per drill (average)")
    
    # Force CPU mode
    device = torch.device('cpu')
    torch.set_num_threads(hardware.get('num_threads', 4))
    torch.set_num_interop_threads(hardware.get('num_threads', 4))
    
    # Prepare models and preprocessor for inference
    # Extract models from all_results and set to CPU
    models = []
    for result in all_results:
        model = result['model'].cpu()  # Ensure CPU
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
    
    # Load preprocessor with saved scaler (CRITICAL: don't refit to avoid data leakage!)
    from inference import load_preprocessor
    
    config = _create_config_from_dict(config_dict)
    results_dir = config_dict['output_paths']['results_dir']
    experiment_name = config_dict['experiment_name']
    scaler_path = os.path.join(results_dir, f"{experiment_name}_scaler.pkl")
    
    preprocessor = load_preprocessor(scaler_path, config)
    
    # Calculate storage usage
    print("\nCalculating storage usage...")
    storage_info = calculate_storage_usage(config_dict, results_dir, experiment_name)
    print(f"  Models: {storage_info['models_mb']:.2f} MB")
    print(f"  Scaler: {storage_info['scaler_mb']:.2f} MB")
    print(f"  Estimated dependencies: {storage_info['estimated_dependencies_gb']:.2f} GB")
    print(f"  Total application: {storage_info['total_application_gb']:.2f} GB")
    
    # Measure peak memory with models loaded
    if PSUTIL_AVAILABLE:
        peak_memory = measure_memory_usage()
        print(f"\nMemory usage:")
        print(f"  Peak memory (models loaded): {peak_memory:.2f} GB")
    else:
        peak_memory = None
        print(f"\nMemory usage: Cannot measure (psutil not available)")
        print("  Install psutil for memory measurements: pip install psutil")
    
    # Get test files
    test_path = config.test_path
    csv_files = glob.glob(os.path.join(test_path, "**", "*.csv"), recursive=True)
    original_files = [f for f in csv_files if not os.path.basename(f).startswith("augmented_")]
    
    num_samples = runtime_config.get('num_samples', 100)
    num_samples = min(num_samples, len(original_files))
    test_files = original_files[:num_samples]
    
    print(f"\nEvaluating runtime on {num_samples} test files...")
    
    class_names = ["Normal", "NPT", "OD"]
    times = []
    memory_samples = []
    
    for csv_file in tqdm(test_files, desc="Measuring runtime"):
        try:
            if PSUTIL_AVAILABLE:
                mem_before = measure_memory_usage()
            
            start_time = time.perf_counter()
            _ = predict_single_drill(csv_file, models, preprocessor, thresholds, 
                                   device=device, class_names=class_names)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            
            if PSUTIL_AVAILABLE:
                mem_after = measure_memory_usage()
                memory_samples.append(mem_after)
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(csv_file)}: {e}")
            continue
    
    times = np.array(times)
    mean_time_seconds = np.mean(times) if len(times) > 0 else 0.0
    
    # Validation
    print("\n" + "="*80)
    print("REQUIREMENT VALIDATION")
    print("="*80)
    
    speed_passed = mean_time_seconds < required_max_time_seconds if len(times) > 0 else False
    storage_passed = storage_info['total_application_gb'] < required_storage_gb
    
    print(f"\n1. Speed Requirement (< {required_max_time_seconds} seconds):")
    print(f"   Average time: {mean_time_seconds:.3f} seconds")
    print(f"   Status: {'✓ PASS' if speed_passed else '✗ FAIL'}")
    
    print(f"\n2. Storage Requirement (< {required_storage_gb}GB for application):")
    print(f"   Application storage: {storage_info['total_application_gb']:.2f} GB")
    print(f"   Status: {'✓ PASS' if storage_passed else '✗ FAIL'}")
    print(f"   Note: {required_storage_gb}GB requirement is for system total (including OS)")
    
    if PSUTIL_AVAILABLE and peak_memory:
        ram_info_available = True
        avg_memory = np.mean(memory_samples) if memory_samples else peak_memory
        print(f"\n3. RAM Usage (measured):")
        print(f"   Peak memory: {peak_memory:.2f} GB")
        if memory_samples:
            print(f"   Average during inference: {avg_memory:.2f} GB")
        print(f"   Note: {required_ram_gb}GB requirement is for system total (including OS)")
        print(f"   Application uses ~{peak_memory:.2f}GB, leaving ~{required_ram_gb - peak_memory:.2f}GB for OS")
    else:
        ram_info_available = False
        print(f"\n3. RAM Usage:")
        print(f"   Cannot measure (install psutil for memory measurements)")
        print(f"   Note: {required_ram_gb}GB requirement is for system total")
    
    all_passed = speed_passed and storage_passed
    print(f"\n" + "="*80)
    print(f"OVERALL STATUS: {'✓ ALL REQUIREMENTS MET' if all_passed else '✗ SOME REQUIREMENTS NOT MET'}")
    print("="*80)
    
    # Detailed statistics
    print(f"\nRuntime Statistics:")
    print(f"  Samples evaluated: {len(times)}")
    if len(times) > 0:
        print(f"  Mean time: {np.mean(times)*1000:.2f} ms ({mean_time_seconds:.3f} seconds)")
        print(f"  Median time: {np.median(times)*1000:.2f} ms")
        print(f"  Std time: {np.std(times)*1000:.2f} ms")
        print(f"  Min time: {np.min(times)*1000:.2f} ms")
        print(f"  Max time: {np.max(times)*1000:.2f} ms")
        print(f"  Throughput: {len(times)/np.sum(times):.2f} drills/second")
    else:
        print(f"  No valid samples processed")
    
    return {
        'mean_time_ms': float(np.mean(times) * 1000) if len(times) > 0 else 0.0,
        'mean_time_seconds': float(mean_time_seconds) if len(times) > 0 else 0.0,
        'median_time_ms': float(np.median(times) * 1000) if len(times) > 0 else 0.0,
        'std_time_ms': float(np.std(times) * 1000) if len(times) > 0 else 0.0,
        'min_time_ms': float(np.min(times) * 1000) if len(times) > 0 else 0.0,
        'max_time_ms': float(np.max(times) * 1000) if len(times) > 0 else 0.0,
        'throughput_drills_per_sec': float(len(times) / np.sum(times)) if len(times) > 0 and np.sum(times) > 0 else 0.0,
        'num_samples': len(times),
        'requirements': {
            'speed_required_seconds': required_max_time_seconds,
            'speed_actual_seconds': float(mean_time_seconds) if len(times) > 0 else 0.0,
            'speed_passed': speed_passed,
            'storage_required_gb': required_storage_gb,
            'storage_actual_gb': storage_info['total_application_gb'],
            'storage_passed': storage_passed,
            'ram_required_gb': required_ram_gb,
            'ram_peak_gb': float(peak_memory) if peak_memory else None,
            'storage_breakdown': storage_info,
            'all_passed': all_passed
        }
    }


def _create_config_from_dict(config_dict):
    """Helper function to create Config from dictionary"""
    from config import Config
    
    config = Config()
    
    # Basic settings
    config.experiment_name = config_dict.get('experiment_name', 'experiment')
    config.seed = config_dict.get('seed', 42)
    
    # Data paths
    data_paths = config_dict.get('data_paths', {})
    config.train_path = data_paths.get('train_path')
    config.test_path = data_paths.get('test_path')
    config.augmented_data_path = data_paths.get('augmented_data_path')
    config.exclude_files_csv = data_paths.get('exclude_files_csv')
    config.data_path = os.path.dirname(data_paths.get('train_path', ''))
    if 'option2_train_path' in data_paths:
        config.option2_train_path = data_paths['option2_train_path']
    
    # Preprocessing
    preprocessing = config_dict.get('preprocessing', {})
    config.max_series_length = preprocessing.get('max_series_length', 10000)
    config.normalize = preprocessing.get('normalize', True)
    config.include_derivatives = preprocessing.get('include_derivatives', False)
    config.validation_split = preprocessing.get('validation_split', 0.2)
    
    # Status mapping
    config.status_mapping = {"Normal": 0, "NPT": 1, "OD": 2}
    config.num_classes = 3
    
    return config

