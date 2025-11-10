"""
Visualize augmented timeseries with individual source file plots below
Shows the augmented timeseries at the top, and below it shows plots for each
original source file that was used to create the augmented timeseries
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob
import json
from pathlib import Path


def plot_timeseries_simple(df, ax1, ax2=None, title='', show_stages=True):
    """
    Plot a simple timeseries with voltage and Z on an existing axis
    
    Args:
        df: DataFrame with 'Voltage', 'Z', and optionally 'Segment'
        ax1: Primary axis (for voltage)
        ax2: Secondary axis (for Z, created if None)
        title: Plot title
        show_stages: Whether to show stage backgrounds
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


def load_metadata(metadata_path):
    """Load metadata JSON file if it exists"""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def find_source_file(source_file_path, data_root):
    """
    Find the actual source file in the data directory
    
    Args:
        source_file_path: Path from metadata (may be relative or absolute)
        data_root: Root directory to search in
    """
    # Normalize the path - handle both absolute and relative paths
    source_file_path = os.path.normpath(source_file_path)
    
    # Try different possible locations (only Train/Normal, Train/NPT, Train/OD - as used in augmentation)
    possible_paths = [
        source_file_path,  # Absolute path (if already fully specified) - try first
        os.path.join(data_root, "Train", "Normal", os.path.basename(source_file_path)),
        os.path.join(data_root, "Train", "NPT", os.path.basename(source_file_path)),
        os.path.join(data_root, "Train", "OD", os.path.basename(source_file_path)),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, search recursively only in Train directories
    train_dir = os.path.join(data_root, "Train")
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            if os.path.basename(source_file_path) in files:
                return os.path.join(root, os.path.basename(source_file_path))
    
    return None


def plot_augmented_with_source_files(augmented_df, source_info, data_root, 
                                     augmented_title='Augmented Timeseries',
                                     save_path=None):
    """
    Create a comprehensive visualization showing:
    1. Augmented timeseries at the top (with source file bars)
    2. Individual plots for each source file below
    
    Args:
        augmented_df: DataFrame of the augmented timeseries
        source_info: List of dicts with source file information
        data_root: Root directory to search for source files
        augmented_title: Title for the augmented plot
        save_path: Path to save the figure
    """
    # Get unique source files
    unique_sources = {}
    for seg_info in source_info:
        source_file = seg_info.get('source_file', 'Unknown')
        if source_file and source_file != 'Unknown':
            if source_file not in unique_sources:
                unique_sources[source_file] = {
                    'segments': [],
                    'stages': set()
                }
            unique_sources[source_file]['segments'].append(seg_info)
            if seg_info.get('stage'):
                unique_sources[source_file]['stages'].add(seg_info.get('stage'))
    
    num_sources = len(unique_sources)
    
    # Create figure with: 1 augmented plot + 1 source info bar + N source plots
    # Height ratios: augmented plot (3), source info (0.5), each source (1)
    total_rows = 2 + num_sources  # augmented + source_info + sources
    height_ratios = [3, 0.5] + [1] * num_sources
    
    fig = plt.figure(figsize=(18, 4 + num_sources * 2))
    gs = fig.add_gridspec(total_rows, 1, height_ratios=height_ratios, hspace=0.15)
    
    # ===== Plot 1: Augmented Timeseries =====
    ax_aug1 = fig.add_subplot(gs[0, 0])
    ax_aug2 = ax_aug1.twinx()
    plot_timeseries_simple(augmented_df, ax_aug1, ax_aug2, 
                          title=augmented_title, show_stages=True)
    ax_aug1.set_xlabel('')  # Remove x-label from top plot
    
    # ===== Plot 2: Source File Information Bar =====
    ax_source = fig.add_subplot(gs[1, 0])
    
    # Create color mapping for source files
    source_colors = {}
    if unique_sources:
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))
        source_colors = {src: colors[i] for i, src in enumerate(unique_sources.keys())}
    
    # Draw source file rectangles
    for seg_info in source_info:
        start = seg_info.get('start_idx', 0)
        end = seg_info.get('end_idx', len(augmented_df))
        source_file = seg_info.get('source_file', 'Unknown')
        
        if source_file and source_file != 'Unknown':
            color = source_colors.get(source_file, '#cccccc')
            filename = os.path.basename(source_file).replace('.csv', '')
            if len(filename) > 20:
                filename = filename[:17] + '...'
            
            rect_width = max(1, end - start)
            rect = mpatches.Rectangle((start, 0), rect_width, 1.0,
                                     facecolor=color, edgecolor='black',
                                     linewidth=1.5, alpha=0.85, zorder=5)
            ax_source.add_patch(rect)
            
            # Add label if segment is large enough
            segment_length_pct = (end - start) / len(augmented_df) * 100
            if segment_length_pct > 3 and (end - start) > 30:
                xm = (start + end) / 2
                ax_source.text(xm, 0.5, filename,
                             ha='center', va='center', fontsize=7,
                             weight='bold', color='black',
                             bbox=dict(boxstyle='round,pad=0.2',
                                     facecolor='white', edgecolor='black', alpha=0.8))
    
    ax_source.set_xlim(0, len(augmented_df))
    ax_source.set_ylim(-0.2, 1.2)
    ax_source.set_ylabel('Source\nFiles', fontsize=9, weight='bold', rotation=0, va='center')
    ax_source.set_xlabel('Sample Index', fontsize=11)
    ax_source.set_yticks([])
    ax_source.grid(True, alpha=0.2, axis='x')
    ax_source.set_title('Source File Segments Used', fontsize=10, pad=5, weight='bold')
    
    # ===== Plot 3+: Individual Source File Plots =====
    source_axes = []
    for idx, (source_file, info) in enumerate(unique_sources.items(), 2):
        # Find the actual source file
        actual_path = find_source_file(source_file, data_root)
        
        if actual_path and os.path.exists(actual_path):
            try:
                source_df = pd.read_csv(actual_path)
                source_title = f"Source {idx-1}: {os.path.basename(source_file).replace('.csv', '')}"
                if len(source_title) > 50:
                    source_title = source_title[:47] + '...'
                
                ax_src1 = fig.add_subplot(gs[idx, 0])
                ax_src2 = ax_src1.twinx()
                plot_timeseries_simple(source_df, ax_src1, ax_src2, 
                                     title=source_title, show_stages=True)
                
                # Add color indicator on the left edge using a thin rectangle
                # Do this after plotting so limits are set
                color = source_colors.get(source_file, '#cccccc')
                y_bottom, y_top = ax_src1.get_ylim()
                x_left, x_right = ax_src1.get_xlim()
                rect_width = max(1, (x_right - x_left) * 0.005)
                rect = mpatches.Rectangle((x_left, y_bottom), 
                                        rect_width,
                                        y_top - y_bottom,
                                        color=color, alpha=0.7, zorder=10)
                ax_src1.add_patch(rect)
                
                source_axes.append((ax_src1, ax_src2))
            except Exception as e:
                # If we can't load the source file, create an empty subplot with error message
                ax_src = fig.add_subplot(gs[idx, 0])
                ax_src.text(0.5, 0.5, 
                           f"Error loading source file:\n{os.path.basename(source_file)}\n{e}",
                           ha='center', va='center', fontsize=10, color='red',
                           transform=ax_src.transAxes)
                ax_src.set_title(f"Source {idx-1}: {os.path.basename(source_file).replace('.csv', '')}",
                                fontsize=11, weight='bold', pad=10)
        else:
            # Source file not found
            ax_src = fig.add_subplot(gs[idx, 0])
            ax_src.text(0.5, 0.5, 
                       f"Source file not found:\n{os.path.basename(source_file)}",
                       ha='center', va='center', fontsize=10, color='red',
                       transform=ax_src.transAxes)
            ax_src.set_title(f"Source {idx-1}: {os.path.basename(source_file).replace('.csv', '')}",
                            fontsize=11, weight='bold', pad=10)
    
    # Final adjustments
    plt.subplots_adjust(hspace=0.15)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_all_augmented_with_sources(data_root="/Users/matteobonsignore/EDM/data/Option 2",
                                        output_dir="/Users/matteobonsignore/EDM/Sequence-Preserving Augmentation/visualizations_with_sources"):
    """
    Generate visualizations for all augmented files with source file plots
    
    Args:
        data_root: Root directory of data
        output_dir: Directory to save visualizations
    """
    train_dir = os.path.join(data_root, "Train")
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['Normal', 'NPT', 'OD']
    
    print("="*70)
    print("Visualizing Sequence-Preserved Augmented Timeseries with Source Files")
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
                print(f"  [{i}/{num_files}] {filename}")
                
                # Load augmented data
                augmented_df = pd.read_csv(csv_file)
                
                # Try to load metadata
                metadata_path = csv_file.replace('.csv', '_metadata.json')
                source_info = load_metadata(metadata_path)
                
                if source_info is None:
                    print(f"    WARNING: No metadata found for {filename}")
                    print(f"    Skipping (cannot show source files without metadata)")
                    continue
                
                print(f"    (Found metadata with {len(source_info)} segments)")
                
                # Create comprehensive visualization
                title = f"{category} - Augmented Timeseries\n{filename}"
                
                output_file = os.path.join(output_category_path, f"{filename}_with_sources.png")
                
                fig = plot_augmented_with_source_files(
                    augmented_df,
                    source_info,
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
    visualize_all_augmented_with_sources(
        data_root="/Users/matteobonsignore/EDM/data/Option 2",
        output_dir="/Users/matteobonsignore/EDM/Sequence-Preserving Augmentation/visualizations_with_sources"
    )

