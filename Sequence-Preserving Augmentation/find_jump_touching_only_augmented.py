"""
Identify augmented OD timeseries where there's an upward jump at Touching phase
in the augmented but NOT in the reference, and move their visualizations
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import shutil


def detect_jump_at_touching_start(df):
    """
    Detect if there's an upward jump at the beginning of a Touching phase
    
    Returns:
        (has_jump, jump_magnitude): True if upward jump detected, and jump amount
    """
    if 'Segment' not in df.columns:
        return False, 0.0
    
    stages = df['Segment'].fillna('').astype(str).values
    z_values = df['Z'].values
    
    # Find ALL Touching phase starts (not just first one)
    touching_starts = []
    in_touching = False
    
    for i in range(len(stages)):
        if stages[i] == 'Touching' and not in_touching:
            # Start of Touching phase
            touching_starts.append(i)
            in_touching = True
        elif stages[i] != 'Touching' and in_touching:
            # End of Touching phase
            in_touching = False
    
    if not touching_starts:
        return False, 0.0
    
    # Check each Touching start for jumps
    max_jump = 0.0
    has_any_jump = False
    
    for touching_start in touching_starts:
        # Check for upward jump at the start of Touching
        # Compare Z values just before and just after the transition
        if touching_start == 0:
            # Touching starts from the beginning, can't check previous
            continue
        
        # Get Z values: last few points before Touching and first few points of Touching
        window = 15  # Increased window for better detection
        before_start = max(0, touching_start - window)
        before_end = touching_start
        
        after_start = touching_start
        after_end = min(len(z_values), touching_start + window)
        
        if after_end <= after_start or before_end <= before_start:
            continue
        
        # Get average Z before and after transition (more robust)
        z_before = np.median(z_values[before_start:before_end])  # Use median for robustness
        z_after = np.median(z_values[after_start:after_end])
        
        # Check for upward jump (positive jump = Z increases, meaning depth decreases)
        jump = z_after - z_before
        
        # Upward jump threshold: at least 5 units upward (lowered to catch more cases)
        if jump > 5.0:
            has_any_jump = True
            max_jump = max(max_jump, jump)
    
    return has_any_jump, max_jump


def load_reference_timeseries(reference_file_path, data_root):
    """Load the reference timeseries file"""
    # Try different possible locations
    reference_file_path = os.path.normpath(reference_file_path)
    
    possible_paths = [
        reference_file_path,  # Absolute path
        os.path.join(data_root, "Train", "Normal", os.path.basename(reference_file_path)),
        os.path.join(data_root, "Train", "NPT", os.path.basename(reference_file_path)),
        os.path.join(data_root, "Train", "OD", os.path.basename(reference_file_path)),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                continue
    
    # Recursive search
    train_dir = os.path.join(data_root, "Train")
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            if os.path.basename(reference_file_path) in files:
                try:
                    return pd.read_csv(os.path.join(root, os.path.basename(reference_file_path)))
                except:
                    continue
    
    return None


def find_jump_touching_only_augmented():
    """
    Find augmented OD timeseries with upward jump at Touching in augmented but not in reference
    """
    data_root = "/Users/matteobonsignore/EDM/data/Option 2"
    viz_dir = "/Users/matteobonsignore/EDM/Sequence-Preserving Augmentation/viz_new/OD"
    output_subdir = os.path.join(viz_dir, "Jump Touch Augm Only")
    
    os.makedirs(output_subdir, exist_ok=True)
    
    train_dir = os.path.join(data_root, "Train")
    od_dir = os.path.join(train_dir, "OD")
    
    # Get all augmented OD files
    csv_files = sorted(glob.glob(os.path.join(od_dir, "augmented_seq_*.csv")))
    
    print("="*70)
    print("Finding Augmented OD Timeseries with Jump at Touching (Augmented Only)")
    print("="*70)
    print(f"\nAnalyzing {len(csv_files)} augmented OD files...\n")
    
    matching_files = []
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            filename = os.path.basename(csv_file).replace('.csv', '')
            if i % 20 == 0:
                print(f"  [{i}/{len(csv_files)}] Checking {filename}...")
            
            # Load augmented timeseries
            augmented_df = pd.read_csv(csv_file)
            
            # Check for jump at Touching in augmented
            has_jump_aug, jump_mag_aug = detect_jump_at_touching_start(augmented_df)
            
            if not has_jump_aug or jump_mag_aug < 5.0:
                continue  # No significant jump in augmented, skip
            
            # Load reference timeseries
            metadata_path = csv_file.replace('.csv', '_metadata.json')
            if not os.path.exists(metadata_path):
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            reference_file = metadata.get('reference_file')
            if not reference_file or reference_file == 'Unknown':
                continue
            
            # Load reference
            ref_df = load_reference_timeseries(reference_file, data_root)
            if ref_df is None:
                continue
            
            # Check for jump at Touching in reference
            has_jump_ref, jump_mag_ref = detect_jump_at_touching_start(ref_df)
            
            # We want: jump in augmented BUT NOT in reference
            # Also allow small jumps in reference (less than 5 units) since we're looking for large jumps only in augmented
            # Match if: augmented has jump > 5 AND reference has no jump or jump < 5
            if jump_mag_aug > 5.0 and (not has_jump_ref or jump_mag_ref < 5.0):
                matching_files.append((csv_file, filename, jump_mag_aug, jump_mag_ref))
                print(f"  ✓ MATCH: {filename}")
                print(f"      Augmented jump: {jump_mag_aug:.2f}, Reference jump: {jump_mag_ref:.2f}")
        
        except Exception as e:
            print(f"    ERROR on {filename}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Found {len(matching_files)} matching files")
    print(f"{'='*70}\n")
    
    # Copy matching visualizations to subfolder
    copied = 0
    for csv_file, filename, jump_aug, jump_ref in matching_files:
        viz_file = os.path.join(viz_dir, f"{filename}_with_reference.png")
        if os.path.exists(viz_file):
            dest_file = os.path.join(output_subdir, f"{filename}_with_reference.png")
            shutil.copy2(viz_file, dest_file)
            copied += 1
    
    print(f"Copied {copied} visualizations to:")
    print(f"  {output_subdir}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    find_jump_touching_only_augmented()

