# Sequence-Preserving Data Augmentation

This project performs data augmentation by **preserving the exact stage sequences** from original timeseries and recreating them using segments from a comprehensive database.

## Approach

1. **Build Segment Database**: Extracts segments from ALL training data in Option 2 (Normal, NPT, OD, MH, etc.)
2. **Analyze Original Patterns**: Identifies unique stage sequences in original timeseries for each label
3. **Preserve Sequences**: Creates augmented timeseries that follow the exact same sequence as originals, but using segments from different source files
4. **Equal Distribution**: Distributes augmentations across different sequence patterns proportionally

## Target Counts

- **Normal**: 220 timeseries (including originals)
- **NPT**: 220 timeseries (including originals)  
- **OD**: 220 timeseries (including originals)

## Key Features

- ✅ Z-value continuity maintained when joining segments
- ✅ All augmented files clearly labeled with `augmented_seq_` prefix
- ✅ Metadata saved for each augmented file showing source segments
- ✅ Visualizations showing source file contributions

## Files

- `sequence_augmentation.py` - Main augmentation script
- `visualize_augmented.py` - Basic visualizations
- `visualize_with_sources.py` - Visualizations with individual source file plots

## Usage

```bash
# Run augmentation
cd "Sequence-Preserving Augmentation"
python sequence_augmentation.py

# Generate basic visualizations
python visualize_augmented.py

# Generate visualizations with source files
python visualize_with_sources.py
```

## Output

- Augmented CSV files saved to: `data/Option 2/Train/{Normal|NPT|OD}/augmented_seq_*.csv`
- Metadata JSON files: `augmented_seq_*_metadata.json`
- Visualizations: `visualizations/` and `visualizations_with_sources/`

