"""
Visualize augmented timeseries with reference timeseries below
Shows the augmented timeseries at the top and its reference (original) timeseries below
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob
import json


def plot_timeseries_simple(df, ax1, ax2=None, title='', show_stages=True):
    """
    Plot a simple timeseries with voltage and Z on an existing axis
    """
    if ax2 is None:
        ax2 = ax1.twinx()
    
    # Stage colors
    stage_colors = {
        'Touching': '#dbeafe',
        'Pre-drilling': '#e6f4ea',
        'Body Drilling': '#fff4e5',
        'Break-through': '#fde8e8',
        'Free Falling': '#ff4444',
        'Rework Free Falling': '#8B0000',
        'Retraction': '#f3e8ff',
        'Scarfing': '#e0f2fe',
        'Abnormal Drill': '#f0f0f0',
        'Machine Health Issue': '#ffcc00'
    }
    
    x = range(len(df))
    voltage = df['Voltage'].values
    z = df['Z'].values
    
    # Validate data
    valid_mask = np.isfinite(voltage) & np.isfinite(z)
    if not np.all(valid_mask):
        voltage = np.array([v if np.isfinite(v) else np.nanmean(voltage[valid_mask]) 
                           for v in voltage])
        z = np.array([v if np.isfinite(v) else np.nanmean(z[valid_mask]) 
                     for v in z])
    
    # Plot stage backgrounds
    if show_stages and 'Segment' in df.columns:
        stages = df['Segment'].fillna('').astype(str)
        change_idx = [0] + list(stages[stages != stages.shift()].index) + [len(df)]
        segments = [(change_idx[i], change_idx[i+1], stages.iloc[change_idx[i]]) 
                   for i in range(len(change_idx)-1)]
        
        for i0, i1, stage in segments:
            if stage and stage != 'nan':
                ax1.axvspan(x[i0], x[i1-1], 
                           facecolor=stage_colors.get(stage, '#eeeeee'), 
                           alpha=0.6, zorder=0, edgecolor='gray', linewidth=0.5)
                # Stage label (only if segment is large enough)
                if (i1 - i0) > len(df) * 0.05:  # Only label if >5% of total
                    xm = 0.5 * (x[i0] + x[i1-1])
                    ax1.text(xm, 0.98, stage, 
                            transform=ax1.get_xaxis_transform(), 
                            ha='center', va='top', fontsize=8, 
                            alpha=0.9, weight='bold')
    
    # Plot signals
    l1, = ax1.plot(x, voltage, linewidth=1.2, label='Voltage', color='#1f77b4', zorder=3)
    l2, = ax2.plot(x, z, linewidth=1.4, linestyle='-', label='Z', color='#ff7f0e', zorder=3)
    
    # Labels
    ax1.set_ylabel('Voltage', color='#1f77b4', fontsize=10)
    ax2.set_ylabel('Z', color='#ff7f0e', fontsize=10)
    ax1.grid(True, alpha=0.25, zorder=1)
    ax1.set_title(title, fontsize=11, weight='bold', pad=10)
    
    # Legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=8)
    
    return ax1, ax2


def find_reference_file(reference_file_path, data_root):
    """Find the reference file in the data directory"""
    # Handle None case
    if not reference_file_path or reference_file_path == 'Unknown':
        return None
    
    # Normalize the path
    reference_file_path = os.path.normpath(reference_file_path)
    
    # Try different possible locations
    possible_paths = [
        reference_file_path,  # Absolute path
        os.path.join(data_root, "Train", "Normal", os.path.basename(reference_file_path)),
        os.path.join(data_root, "Train", "NPT", os.path.basename(reference_file_path)),
        os.path.join(data_root, "Train", "OD", os.path.basename(reference_file_path)),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Recursive search in Train directory
    train_dir = os.path.join(data_root, "Train")
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            if os.path.basename(reference_file_path) in files:
                return os.path.join(root, os.path.basename(reference_file_path))
    
    return None


def plot_augmented_with_reference(augmented_df, reference_file_path, data_root,
                                  augmented_title='Augmented Timeseries',
                                  save_path=None):
    """
    Create visualization showing:
    1. Augmented timeseries at the top
    2. Reference timeseries below it
    """
    # Find the reference file
    actual_ref_path = find_reference_file(reference_file_path, data_root)
    
    if not actual_ref_path or not os.path.exists(actual_ref_path):
        # Create figure with just augmented plot and error message
        ref_msg = ""
        if reference_file_path:
            ref_msg = f"\n(Reference file not found: {os.path.basename(reference_file_path)})"
        else:
            ref_msg = "\n(No reference file specified)"
        
        fig = plt.figure(figsize=(18, 6))
        ax_aug1 = fig.add_subplot(1, 1, 1)
        ax_aug2 = ax_aug1.twinx()
        plot_timeseries_simple(augmented_df, ax_aug1, ax_aug2, 
                              title=f"{augmented_title}{ref_msg}",
                              show_stages=True)
        ax_aug1.set_xlabel('Sample Index', fontsize=11)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved (without reference): {save_path}")
        return fig
    
    # Load reference timeseries
    try:
        ref_df = pd.read_csv(actual_ref_path)
    except Exception as e:
        print(f"Error loading reference file {actual_ref_path}: {e}")
        # Create figure with just augmented plot
        fig = plt.figure(figsize=(18, 6))
        ax_aug1 = fig.add_subplot(1, 1, 1)
        ax_aug2 = ax_aug1.twinx()
        plot_timeseries_simple(augmented_df, ax_aug1, ax_aug2, 
                              title=f"{augmented_title}\n(Error loading reference)",
                              show_stages=True)
        ax_aug1.set_xlabel('Sample Index', fontsize=11)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Create figure with two subplots: augmented (top) + reference (bottom)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.15)
    
    # ===== Top Plot: Augmented Timeseries =====
    ax_aug1 = fig.add_subplot(gs[0, 0])
    ax_aug2 = ax_aug1.twinx()
    plot_timeseries_simple(augmented_df, ax_aug1, ax_aug2, 
                          title=augmented_title, show_stages=True)
    ax_aug1.set_xlabel('')  # Remove x-label from top plot
    
    # ===== Bottom Plot: Reference Timeseries =====
    ref_filename = os.path.basename(actual_ref_path).replace('.csv', '')
    if len(ref_filename) > 50:
        ref_filename = ref_filename[:47] + '...'
    ref_title = f"Reference Timeseries: {ref_filename}"
    
    ax_ref1 = fig.add_subplot(gs[1, 0])
    ax_ref2 = ax_ref1.twinx()
    plot_timeseries_simple(ref_df, ax_ref1, ax_ref2, 
                          title=ref_title, show_stages=True)
    ax_ref1.set_xlabel('Sample Index', fontsize=11)
    
    # Final adjustments
    plt.subplots_adjust(hspace=0.15)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def load_metadata(metadata_path):
    """Load metadata JSON file if it exists"""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def visualize_all_augmented_with_reference(data_root="/Users/matteobonsignore/EDM/data/Option 2",
                                          output_dir="viz_new"):
    """
    Generate visualizations for all augmented files with reference timeseries
    """
    train_dir = os.path.join(data_root, "Train")
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['Normal', 'NPT', 'OD']
    
    print("="*70)
    print("Visualizing Augmented Timeseries with Reference")
    print("="*70)
    
    for category in categories:
        print(f"\nProcessing {category}...")
        category_path = os.path.join(train_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        # Get ALL augmented files (sequence-preserved ones)
        csv_files = sorted(glob.glob(os.path.join(category_path, "augmented_seq_*.csv")))
        
        num_files = len(csv_files)
        if num_files == 0:
            print(f"  No augmented files found in {category}")
            continue
        
        print(f"  Found {num_files} augmented files to visualize")
        
        for i, csv_file in enumerate(csv_files, 1):
            try:
                filename = os.path.basename(csv_file).replace('.csv', '')
                if i % 50 == 0 or i == 1:
                    print(f"  [{i}/{num_files}] {filename}")
                
                # Load augmented data
                augmented_df = pd.read_csv(csv_file)
                
                # Try to load metadata to get reference file
                metadata_path = csv_file.replace('.csv', '_metadata.json')
                metadata = load_metadata(metadata_path)
                
                reference_file = None
                if metadata:
                    # New metadata structure has 'reference_file' at top level
                    if isinstance(metadata, dict) and 'reference_file' in metadata:
                        reference_file = metadata['reference_file']
                        if reference_file and reference_file != 'Unknown':
                            # Extract just the filename if it's a full path
                            if os.path.sep in str(reference_file):
                                reference_file = reference_file
                            else:
                                # It's just a filename, we'll search for it
                                reference_file = reference_file
                    # Old structure would be a list - not used in new format
                
                if not reference_file or reference_file == 'Unknown':
                    if i <= 5:  # Only print warning for first few
                        print(f"    WARNING: No reference file found for {filename}")
                    reference_file = None
                
                # Create visualization
                title = f"{category} - Augmented Timeseries\n{filename}"
                
                output_file = os.path.join(output_category_path, f"{filename}_with_reference.png")
                
                fig = plot_augmented_with_reference(
                    augmented_df,
                    reference_file,
                    data_root,
                    augmented_title=title,
                    save_path=output_file
                )
                
                plt.close(fig)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*70}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    visualize_all_augmented_with_reference(
        data_root="/Users/matteobonsignore/EDM/data/Option 2",
        output_dir="/Users/matteobonsignore/EDM/Sequence-Preserving Augmentation/viz_new"
    )

