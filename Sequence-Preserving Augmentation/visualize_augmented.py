"""
Visualize sequence-preserved augmented timeseries
Adapted from the original visualization script
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob
import json


def plot_augmented_timeseries_with_sources(df, source_info=None, title='Augmented Timeseries', 
                                          save_path=None):
    """
    Plot augmented timeseries showing:
    1. Voltage and Z signals (overlaid, like original)
    2. Stage segments with colors
    3. Source file information for each segment (colored bars, separate subplot)
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax_source = fig.add_subplot(gs[1])
    
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
    if 'Segment' in df.columns:
        stages = df['Segment'].fillna('').astype(str)
        change_idx = [0] + list(stages[stages != stages.shift()].index) + [len(df)]
        segments = [(change_idx[i], change_idx[i+1], stages.iloc[change_idx[i]]) 
                   for i in range(len(change_idx)-1)]
        
        for i0, i1, stage in segments:
            if stage and stage != 'nan':
                ax1.axvspan(x[i0], x[i1-1], 
                           facecolor=stage_colors.get(stage, '#eeeeee'), 
                           alpha=0.6, zorder=0, edgecolor='gray', linewidth=0.5)
                xm = 0.5 * (x[i0] + x[i1-1])
                ax1.text(xm, 0.98, stage, 
                        transform=ax1.get_xaxis_transform(), 
                        ha='center', va='top', fontsize=9, 
                        alpha=0.9, weight='bold')
    
    # Plot signals
    l1, = ax1.plot(x, voltage, linewidth=1.2, label='Voltage', color='#1f77b4', zorder=3)
    l2, = ax2.plot(x, z, linewidth=1.4, linestyle='-', label='Z', color='#ff7f0e', zorder=3)
    
    # Labels
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Voltage', color='#1f77b4', fontsize=12)
    ax2.set_ylabel('Z (Depth)', color='#ff7f0e', fontsize=12)
    ax1.grid(True, alpha=0.25, zorder=1)
    ax1.set_title(title, fontsize=14, weight='bold', pad=15)
    
    # Main legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    legend1 = ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    # Source file information
    source_color_map = {}
    source_summary = {}
    
    # If source_info is None, try to infer from stage segments
    if source_info is None:
        source_info = []
        if 'Segment' in df.columns:
            stages = df['Segment'].fillna('').astype(str)
            change_idx = [0] + list(stages[stages != stages.shift()].index) + [len(df)]
            for i in range(len(change_idx)-1):
                source_info.append({
                    'start_idx': change_idx[i],
                    'end_idx': change_idx[i+1],
                    'stage': stages.iloc[change_idx[i]],
                    'source_file': 'Unknown'
                })
    
    if source_info and len(source_info) > 0 and ax_source:
        unique_sources = list(set([s.get('source_file', 'Unknown') 
                                  for s in source_info 
                                  if s.get('source_file') and s.get('source_file') != 'Unknown']))
        
        if unique_sources:
            source_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))
            source_color_map = {src: source_colors[i] for i, src in enumerate(unique_sources)}
        
        y_height = 1.0
        
        for seg_info in source_info:
            start = seg_info.get('start_idx', 0)
            end = seg_info.get('end_idx', len(df))
            source_file = seg_info.get('source_file', 'Unknown')
            stage = seg_info.get('stage', '')
            
            if source_file and source_file != 'Unknown':
                filename = os.path.basename(source_file).replace('.csv', '')
                if len(filename) > 18:
                    filename = filename[:15] + '...'
                
                color = source_color_map.get(source_file, '#cccccc')
                
                rect_width = max(1, end - start)
                rect = mpatches.Rectangle((start, 0), rect_width, 1.0,
                                         facecolor=color, edgecolor='black',
                                         linewidth=1.5, alpha=0.85, zorder=5)
                ax_source.add_patch(rect)
                
                xm = (start + end) / 2
                segment_length_pct = (end - start) / len(df) * 100
                
                if segment_length_pct > 2 and (end - start) > 30:
                    ax_source.text(xm, 0.5, filename,
                                 ha='center', va='center', fontsize=7,
                                 weight='bold', color='black',
                                 bbox=dict(boxstyle='round,pad=0.2',
                                         facecolor='white', edgecolor='black', alpha=0.8))
                
                if source_file not in source_summary:
                    source_summary[source_file] = {'count': 0, 'stages': set()}
                source_summary[source_file]['count'] += 1
                if stage:
                    source_summary[source_file]['stages'].add(stage)
        
        ax_source.set_xlim(0, len(df))
        ax_source.set_ylim(-0.5, 1.5)
        ax_source.set_ylabel('Source\nFiles', fontsize=9, weight='bold', rotation=0, va='center')
        ax_source.set_xlabel('Sample Index', fontsize=11)
        ax_source.set_yticks([])
        ax_source.grid(True, alpha=0.2, axis='x')
        ax_source.set_title('Source Files Used', fontsize=10, pad=5, weight='bold')
        ax_source.sharex(ax1)
        ax_source.set_xlim(0, len(df))
        
        if source_summary:
            legend_elements = []
            for src, info in list(source_summary.items())[:8]:
                filename = os.path.basename(src).replace('.csv', '')
                if len(filename) > 20:
                    filename = filename[:17] + '...'
                
                stages_list = list(info['stages'])
                if stages_list:
                    stages_str = ', '.join(stages_list[:2])
                    if len(stages_list) > 2:
                        stages_str += f' (+{len(stages_list)-2})'
                    label = f"{filename} ({info['count']} seg: {stages_str})"
                else:
                    label = f"{filename} ({info['count']} seg)"
                
                legend_elements.append(
                    mpatches.Patch(facecolor=source_color_map.get(src, '#cccccc'),
                                 edgecolor='black', label=label)
                )
            
            if len(source_summary) > 8:
                legend_elements.append(
                    mpatches.Patch(facecolor='#cccccc', edgecolor='black',
                                 label=f'+ {len(source_summary) - 8} more sources...')
                )
            
            if legend_elements:
                legend2 = ax1.legend(legend_elements, 
                          [e.get_label() for e in legend_elements],
                          loc='upper right', fontsize=7, framealpha=0.95, 
                          title='Source Files', title_fontsize=8)
                ax1.add_artist(legend1)
    
    # Always configure source subplot
    if ax_source:
        ax_source.set_xlim(0, len(df))
        ax_source.set_ylim(-0.5, 1.5)
        if not source_info or len(source_info) == 0:
            ax_source.text(len(df)/2, 0.5, 'No source file information available',
                          ha='center', va='center', fontsize=10, 
                          style='italic', color='gray')
        ax_source.set_ylabel('Source\nFiles', fontsize=9, weight='bold', rotation=0, va='center')
        ax_source.set_xlabel('Sample Index', fontsize=11)
        ax_source.set_yticks([])
        ax_source.grid(True, alpha=0.2, axis='x')
        ax_source.set_title('Source Files Used', fontsize=10, pad=5, weight='bold')
    
    plt.subplots_adjust(hspace=0.05)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def load_metadata(metadata_path):
    """Load metadata JSON file if it exists"""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def visualize_all_augmented(data_root="/Users/matteobonsignore/EDM/data/Option 2",
                            output_dir="visualizations"):
    """
    Generate visualizations for all augmented files
    """
    train_dir = os.path.join(data_root, "Train")
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['Normal', 'NPT', 'OD']
    
    print("="*70)
    print("Visualizing Sequence-Preserved Augmented Timeseries")
    print("="*70)
    
    for category in categories:
        print(f"\nProcessing {category}...")
        category_path = os.path.join(train_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        csv_files = sorted(glob.glob(os.path.join(category_path, "augmented_seq_*.csv")))
        
        num_files = len(csv_files)
        if num_files == 0:
            print(f"  No augmented files found in {category}")
            continue
        
        print(f"  Found {num_files} augmented files to visualize")
        
        for i, csv_file in enumerate(csv_files, 1):
            try:
                filename = os.path.basename(csv_file).replace('.csv', '')
                if i % 50 == 0:
                    print(f"  [{i}/{num_files}] {filename}")
                
                df = pd.read_csv(csv_file)
                
                metadata_path = csv_file.replace('.csv', '_metadata.json')
                source_info = load_metadata(metadata_path)
                
                title = f"{category} - Sequence-Preserved Augmented\n{filename}"
                
                output_file = os.path.join(output_category_path, f"{filename}.png")
                
                fig, _ = plot_augmented_timeseries_with_sources(
                    df, 
                    source_info=source_info,
                    title=title,
                    save_path=output_file
                )
                
                plt.close(fig)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
    
    print(f"\n{'='*70}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    visualize_all_augmented(
        data_root="/Users/matteobonsignore/EDM/data/Option 2",
        output_dir="/Users/matteobonsignore/EDM/Sequence-Preserving Augmentation/visualizations"
    )

