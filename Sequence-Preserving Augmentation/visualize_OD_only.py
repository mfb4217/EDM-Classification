"""
Generate visualizations for OD files only (to complete the missing ones)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from visualize_with_reference import *

def visualize_OD_only():
    """Generate visualizations for OD category only"""
    data_root = "/Users/matteobonsignore/EDM/data/Option 2"
    output_dir = "/Users/matteobonsignore/EDM/Sequence-Preserving Augmentation/viz_new"
    
    train_dir = os.path.join(data_root, "Train")
    category = "OD"
    
    print("="*70)
    print(f"Visualizing OD Augmented Timeseries with Reference")
    print("="*70)
    
    category_path = os.path.join(train_dir, category)
    output_category_path = os.path.join(output_dir, category)
    os.makedirs(output_category_path, exist_ok=True)
    
    # Get ALL augmented files
    csv_files = sorted(glob.glob(os.path.join(category_path, "augmented_seq_*.csv")))
    
    # Get already created visualizations
    existing_viz = set()
    if os.path.exists(output_category_path):
        for f in os.listdir(output_category_path):
            if f.endswith('_with_reference.png'):
                # Extract the base filename
                base = f.replace('_with_reference.png', '')
                existing_viz.add(base)
    
    # Filter to only process missing ones
    to_process = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).replace('.csv', '')
        if filename not in existing_viz:
            to_process.append((csv_file, filename))
    
    num_files = len(csv_files)
    num_to_process = len(to_process)
    num_done = num_files - num_to_process
    
    print(f"\nFound {num_files} OD augmented files")
    print(f"  Already visualized: {num_done}")
    print(f"  To process: {num_to_process}")
    
    for idx, (csv_file, filename) in enumerate(to_process, 1):
        try:
            if idx % 10 == 0 or idx == 1:
                print(f"  [{idx}/{num_to_process}] {filename}")
            
            # Load augmented data
            augmented_df = pd.read_csv(csv_file)
            
            # Load metadata
            metadata_path = csv_file.replace('.csv', '_metadata.json')
            metadata = load_metadata(metadata_path)
            
            reference_file = None
            if metadata and isinstance(metadata, dict) and 'reference_file' in metadata:
                reference_file = metadata['reference_file']
            
            if not reference_file or reference_file == 'Unknown':
                reference_file = None
            
            # Create visualization
            title = f"OD - Augmented Timeseries\n{filename}"
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
            print(f"    ERROR on {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    final_count = len([f for f in os.listdir(output_category_path) if f.endswith('.png')])
    print(f"OD visualizations complete: {final_count}/{num_files}")
    print(f"{'='*70}")

if __name__ == "__main__":
    visualize_OD_only()

