"""
Sequence-Preserving Data Augmentation for EDM Timeseries
Extracts segments from all training data and recreates exact stage sequences
from original timeseries using segments from a database.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import uuid
from typing import List, Tuple, Dict, Optional
import json
from collections import defaultdict
import random
from dataclasses import dataclass


@dataclass
class TimeseriesSegment:
    """Represents a segment of a timeseries"""
    data: pd.DataFrame
    stage: str
    source_file: str
    
    @property
    def label(self) -> str:
        """Infer dataset label (e.g., NPT, OD) from the source file path"""
        normalized_path = os.path.normpath(self.source_file)
        parts = normalized_path.split(os.sep)
        if 'Train' in parts:
            train_idx = parts.index('Train')
            if train_idx + 1 < len(parts):
                return parts[train_idx + 1]
        return ""
    
    @property
    def start_z(self) -> float:
        return self.data['Z'].iloc[0] if len(self.data) > 0 else 0
    
    @property
    def end_z(self) -> float:
        return self.data['Z'].iloc[-1] if len(self.data) > 0 else 0
    
    def __len__(self):
        return len(self.data)


class SequencePreservingAugmenter:
    """Augments timeseries by preserving exact stage sequences from originals"""
    
    def __init__(self, data_root: str, seed: int = 42):
        self.data_root = data_root
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Segment database: stage -> list of segments
        self.segment_database: Dict[str, List[TimeseriesSegment]] = defaultdict(list)
        
        # Original timeseries analyzed: (label, sequence) -> list of file paths
        self.original_patterns: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    
    def load_timeseries(self, file_path: str) -> pd.DataFrame:
        """Load a timeseries CSV file"""
        return pd.read_csv(file_path)
    
    def extract_sequence(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Extract the sequence of stages from a timeseries
        
        Returns:
            (sequence_string, stage_list): e.g., ("Touching_BodyDrilling_Breakthrough", [...])
        """
        stages = []
        current_stage = None
        
        for _, row in df.iterrows():
            stage = row['Segment']
            if stage != current_stage:
                if stage not in ['', 'nan', None]:
                    stages.append(stage)
                current_stage = stage
        
        # Create a unique sequence identifier
        sequence_str = '_'.join(stages)
        return sequence_str, stages
    
    def extract_segments(self, df: pd.DataFrame, source_file: str) -> List[TimeseriesSegment]:
        """Extract all segments from a timeseries"""
        segments = []
        current_stage = None
        current_segment_data = []
        
        for idx, row in df.iterrows():
            stage = row['Segment']
            
            if stage != current_stage:
                # Save previous segment if it exists
                if current_stage is not None and len(current_segment_data) > 0:
                    segment_df = pd.DataFrame(current_segment_data, 
                                             columns=['Voltage', 'Z', 'Segment'])
                    segments.append(TimeseriesSegment(segment_df, current_stage, source_file))
                
                # Start new segment
                current_stage = stage
                current_segment_data = []
            
            if stage not in ['', 'nan', None]:
                current_segment_data.append([row['Voltage'], row['Z'], row['Segment']])
        
        # Add final segment
        if current_stage is not None and len(current_segment_data) > 0:
            segment_df = pd.DataFrame(current_segment_data, 
                                     columns=['Voltage', 'Z', 'Segment'])
            segments.append(TimeseriesSegment(segment_df, current_stage, source_file))
        
        return segments
    
    def build_segment_database(self, train_dir: str):
        """
        Build a database of segments from ALL training data in Option 2
        Excludes segments with Machine Health Issue stage
        
        Args:
            train_dir: Path to Train directory
        """
        print("="*70)
        print("Building Segment Database from All Training Data")
        print("Excluding Machine Health Issue segments")
        print("="*70)
        
        # Get all subdirectories (Normal, NPT, OD, MH, etc.)
        categories = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
        
        total_segments = 0
        excluded_segments = 0
        
        for category in categories:
            category_path = os.path.join(train_dir, category)
            csv_files = glob.glob(os.path.join(category_path, "*.csv"))
            
            # Filter out augmented files
            original_files = [f for f in csv_files 
                            if not os.path.basename(f).startswith("augmented_")]
            
            print(f"\nProcessing {category}: {len(original_files)} files")
            
            for csv_file in original_files:
                try:
                    df = self.load_timeseries(csv_file)
                    segments = self.extract_segments(df, csv_file)
                    
                    for segment in segments:
                        # Exclude Machine Health Issue segments
                        if segment.stage == 'Machine Health Issue':
                            excluded_segments += 1
                            continue
                        
                        self.segment_database[segment.stage].append(segment)
                        total_segments += 1
                except Exception as e:
                    print(f"  Error processing {os.path.basename(csv_file)}: {e}")
                    continue
        
        print(f"\n{'='*70}")
        print(f"Segment Database Summary:")
        for stage, segments in sorted(self.segment_database.items()):
            print(f"  {stage}: {len(segments)} segments")
        print(f"  Total segments: {total_segments}")
        if excluded_segments > 0:
            print(f"  Excluded {excluded_segments} Machine Health Issue segments")
        print(f"{'='*70}")
    
    def analyze_original_timeseries(self, train_dir: str, target_labels: List[str]):
        """
        Analyze original timeseries to identify their stage sequences
        Excludes any timeseries with Machine Health Issue stages
        
        Args:
            train_dir: Path to Train directory
            target_labels: List of labels to analyze (e.g., ['Normal', 'NPT', 'OD'])
        """
        print("\n" + "="*70)
        print("Analyzing Original Timeseries Patterns")
        print("Excluding timeseries with Machine Health Issue stages")
        print("="*70)
        
        excluded_count = 0
        
        for label in target_labels:
            label_path = os.path.join(train_dir, label)
            if not os.path.exists(label_path):
                continue
            
            csv_files = glob.glob(os.path.join(label_path, "*.csv"))
            original_files = [f for f in csv_files 
                            if not os.path.basename(f).startswith("augmented_")]
            
            print(f"\n{label}: {len(original_files)} files")
            
            for csv_file in original_files:
                try:
                    df = self.load_timeseries(csv_file)
                    sequence_str, stages = self.extract_sequence(df)
                    
                    # Exclude if sequence contains Machine Health Issue
                    if 'Machine Health Issue' in stages:
                        excluded_count += 1
                        continue
                    
                    if sequence_str:  # Only record if we found stages
                        pattern_key = (label, sequence_str)
                        self.original_patterns[pattern_key].append(csv_file)
                except Exception as e:
                    print(f"  Error analyzing {os.path.basename(csv_file)}: {e}")
                    continue
        
        if excluded_count > 0:
            print(f"\n  Excluded {excluded_count} timeseries containing Machine Health Issue stages")
        
        print(f"\n{'='*70}")
        print("Pattern Summary:")
        for (label, sequence), files in sorted(self.original_patterns.items()):
            print(f"  {label}: {sequence[:50]}... ({len(files)} files)")
        print(f"{'='*70}")
    
    def ensure_z_continuity(self, prev_end_z: float, next_start_z: float, 
                           reference_jump: Optional[float] = None) -> float:
        """
        Calculate Z offset needed to ensure continuity, optionally preserving reference jump
        
        Args:
            prev_end_z: Z value at end of previous segment
            next_start_z: Z value at start of next segment (before offset)
            reference_jump: If provided, preserve this jump pattern from reference timeseries
        
        Returns:
            Z offset to apply
        """
        if reference_jump is not None:
            # Preserve the jump pattern from reference
            # reference_jump = ref_end_z - ref_start_z (could be positive for upward jump)
            # If reference_jump is 0.0, ensure continuity (no jump)
            if reference_jump == 0.0:
                # Ensure continuity: make next segment start at prev_end_z
                offset = prev_end_z - next_start_z
                return offset
            else:
                # Preserve the jump pattern from reference
                # We want: (next_start_z + offset) - prev_end_z = reference_jump
                offset = prev_end_z - next_start_z + reference_jump
                return offset
        
        # Original logic for continuity (when no reference jump specified)
        tolerance = 5.0
        gap = next_start_z - prev_end_z
        
        if abs(gap) > tolerance:
            # Calculate offset to make next segment start close to prev_end_z
            # Add a small increment to maintain progression (drilling goes deeper)
            offset = prev_end_z - next_start_z + 1.0
            return offset
        
        return 0.0
    
    def analyze_transition_jumps(self, df: pd.DataFrame, sequence: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Analyze Z-value jumps at stage transitions in a reference timeseries
        
        Args:
            df: Reference timeseries DataFrame
            sequence: List of stages in order
        
        Returns:
            Dict mapping (stage1, stage2) -> jump_amount (positive = upward)
        """
        jumps = {}
        
        # Get stage boundaries
        stages = df['Segment'].fillna('').astype(str).values
        z_values = df['Z'].values
        
        current_stage_idx = 0
        for i in range(len(stages) - 1):
            if stages[i] != stages[i+1] and stages[i] in sequence and stages[i+1] in sequence:
                # Transition detected
                stage1 = stages[i]
                stage2 = stages[i+1]
                
                # Get Z values at transition
                # Find last Z value of stage1 and first Z value of stage2
                # Look backwards and forwards to get stable values
                stage1_end_idx = i
                stage2_start_idx = i + 1
                
                # Get average of last few points of stage1 and first few points of stage2
                window = 5
                stage1_end_z = np.mean(z_values[max(0, stage1_end_idx-window+1):stage1_end_idx+1])
                stage2_start_z = np.mean(z_values[stage2_start_idx:min(len(z_values), stage2_start_idx+window)])
                
                jump = stage2_start_z - stage1_end_z  # Positive = upward jump
                jumps[(stage1, stage2)] = jump
        
        return jumps
    
    def join_segments(self, segments: List[TimeseriesSegment], 
                     transition_jumps: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Join segments while maintaining Z-continuity and preserving reference jumps
        
        Args:
            segments: List of segments to join
            transition_jumps: Dict of (stage1, stage2) -> jump_amount from reference timeseries
        
        Returns:
            (combined_df, metadata): Combined dataframe and metadata about segments
        """
        if not segments:
            return pd.DataFrame(), []
        
        combined_data = []
        metadata = []
        current_z_offset = 0.0
        
        for i, segment in enumerate(segments):
            # Apply Z offset
            segment_data = segment.data.copy()
            if i > 0:
                # Calculate offset for continuity
                prev_end_z = combined_data[-1]['Z'] if combined_data else 0
                current_start_z = segment.start_z
                
                # Check if we have a reference jump pattern for this transition
                prev_stage = segments[i-1].stage
                curr_stage = segment.stage
                transition_key = (prev_stage, curr_stage)
                
                reference_jump = None
                if transition_jumps and transition_key in transition_jumps:
                    jump_value = transition_jumps[transition_key]
                    
                    # Special handling for Touching phase transitions
                    # If transitioning TO Touching and reference has no significant jump, 
                    # ensure continuity without creating artificial jumps
                    if curr_stage == 'Touching':
                        if abs(jump_value) < 5.0:
                            # Small or no jump in reference - ensure continuity (no jump)
                            reference_jump = 0.0
                        else:
                            # Significant jump in reference - preserve it
                            reference_jump = jump_value
                    else:
                        # Not a Touching transition - preserve the jump pattern
                        reference_jump = jump_value
                elif curr_stage == 'Touching':
                    # No jump recorded for this Touching transition in reference - ensure continuity
                    reference_jump = 0.0
                
                z_offset = self.ensure_z_continuity(
                    prev_end_z, 
                    current_start_z + current_z_offset,
                    reference_jump=reference_jump
                )
                current_z_offset += z_offset
            
            segment_data['Z'] = segment_data['Z'] + current_z_offset
            
            # Record metadata
            start_idx = len(combined_data)
            combined_data.extend(segment_data.to_dict('records'))
            end_idx = len(combined_data)
            
            metadata.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'source_file': segment.source_file,
                'stage': segment.stage,
                'original_length': len(segment)
            })
        
        combined_df = pd.DataFrame(combined_data)
        return combined_df, metadata
    
    def create_augmented_from_sequence(self, sequence: List[str], 
                                      reference_file: Optional[str] = None,
                                      num_augmentations: int = 1,
                                      min_length: int = 500,
                                      target_label: Optional[str] = None) -> List[Tuple[pd.DataFrame, List[Dict], str]]:
        """
        Create augmented timeseries following a specific stage sequence, preserving reference jumps
        
        Args:
            sequence: List of stage names in order (e.g., ['Touching', 'Body Drilling', ...])
            reference_file: Path to reference timeseries file (optional, for preserving jumps)
            num_augmentations: Number of augmented versions to create
            min_length: Minimum length for valid timeseries
            target_label: Dataset label the augmentation belongs to (e.g., 'NPT')
        
        Returns:
            List of (dataframe, metadata, reference_file) tuples
        """
        results = []
        
        # Check that we have segments for all stages in the sequence
        # Also verify no Machine Health Issue stages in sequence
        if 'Machine Health Issue' in sequence:
            print(f"  WARNING: Sequence contains Machine Health Issue - skipping")
            return []
        
        for stage in sequence:
            if stage not in self.segment_database or len(self.segment_database[stage]) == 0:
                print(f"  WARNING: No segments found for stage '{stage}' in database")
                return []
        
        # Analyze reference timeseries for jump patterns if provided
        transition_jumps = None
        if reference_file and os.path.exists(reference_file):
            try:
                ref_df = self.load_timeseries(reference_file)
                transition_jumps = self.analyze_transition_jumps(ref_df, sequence)
            except Exception as e:
                print(f"    Warning: Could not analyze reference jumps from {reference_file}: {e}")
                transition_jumps = None
        
        for _ in range(num_augmentations):
            try:
                # For each stage in sequence, randomly select a segment
                selected_segments = []
                for idx, stage in enumerate(sequence):
                    available_segments = self.segment_database[stage]
                    if target_label == 'NPT' and idx == len(sequence) - 1:
                        npt_segments = [seg for seg in available_segments if seg.label == 'NPT']
                        if npt_segments:
                            available_segments = npt_segments
                        else:
                            print("    Warning: No NPT-origin segments available for final stage; using general pool")
                    if available_segments:
                        selected_segment = random.choice(available_segments)
                        selected_segments.append(selected_segment)
                
                if len(selected_segments) == len(sequence):
                    # Join segments (with jump preservation if available)
                    combined_df, metadata = self.join_segments(selected_segments, transition_jumps)
                    
                    if len(combined_df) >= min_length:
                        # Add reference file to metadata
                        results.append((combined_df, metadata, reference_file or 'Unknown'))
                    
            except Exception as e:
                print(f"    Error creating augmentation: {e}")
                continue
        
        return results
    
    def augment_dataset(self, train_dir: str, target_counts: Dict[str, int]):
        """
        Augment dataset to reach target counts while preserving sequences
        
        Args:
            train_dir: Path to Train directory
            target_counts: Target counts per label, e.g., {'Normal': 220, 'NPT': 220, 'OD': 220}
        """
        print("\n" + "="*70)
        print("Sequence-Preserving Data Augmentation")
        print("="*70)
        
        # Count existing files
        existing_counts = {}
        for label in target_counts.keys():
            label_path = os.path.join(train_dir, label)
            if os.path.exists(label_path):
                csv_files = glob.glob(os.path.join(label_path, "*.csv"))
                original_files = [f for f in csv_files 
                                if not os.path.basename(f).startswith("augmented_")]
                existing_counts[label] = len(original_files)
            else:
                existing_counts[label] = 0
        
        print("\nCurrent counts:")
        for label in target_counts.keys():
            need = max(0, target_counts[label] - existing_counts[label])
            print(f"  {label}: {existing_counts[label]} existing, need {need} more")
        
        # For each label, process each unique sequence pattern
        for label in target_counts.keys():
            print(f"\n{'='*70}")
            print(f"Augmenting {label}")
            print("="*70)
            
            # Get all patterns for this label
            label_patterns = {seq: files for (lbl, seq), files in self.original_patterns.items() 
                           if lbl == label}
            
            if not label_patterns:
                print(f"  No patterns found for {label}")
                continue
            
            # Calculate how many augmentations per pattern
            need_total = max(0, target_counts[label] - existing_counts[label])
            
            if need_total == 0:
                print(f"  Already have {existing_counts[label]} files, no augmentation needed")
                continue
            
            # Count total files across all patterns
            total_originals = sum(len(files) for files in label_patterns.values())
            
            # Distribute augmentations across patterns proportionally
            augmentations_created = 0
            pattern_results = {}
            
            for sequence_str, original_files in label_patterns.items():
                # Calculate how many augmentations for this pattern
                pattern_weight = len(original_files) / total_originals if total_originals > 0 else 0
                num_for_pattern = max(1, int(need_total * pattern_weight))
                
                # Extract stage list from sequence string
                stages = sequence_str.split('_')
                
                print(f"\n  Pattern: {sequence_str[:60]}...")
                print(f"    Original files with this pattern: {len(original_files)}")
                print(f"    Creating {num_for_pattern} augmentations")
                
                # Distribute augmentations across reference files
                # Each augmentation uses one of the original files as reference for jump preservation
                all_results = []
                augmentations_per_ref = max(1, num_for_pattern // len(original_files))
                remaining = num_for_pattern
                
                for ref_file in original_files:
                    if remaining <= 0:
                        break
                    
                    # Create augmentations using this reference file
                    num_from_this_ref = min(augmentations_per_ref, remaining)
                    results = self.create_augmented_from_sequence(
                        stages,
                        reference_file=ref_file,  # Pass reference file for jump preservation
                        num_augmentations=num_from_this_ref,
                        min_length=500,
                        target_label=label
                    )
                    
                    all_results.extend(results)
                    remaining -= len(results)
                
                # If we still need more, create more with random references
                if remaining > 0:
                    for _ in range(min(remaining, 10)):
                        ref_file = random.choice(original_files)
                        results = self.create_augmented_from_sequence(
                            stages,
                            reference_file=ref_file,
                            num_augmentations=1,
                            min_length=500,
                            target_label=label
                        )
                        all_results.extend(results)
                        remaining -= len(results)
                        if remaining <= 0:
                            break
                
                pattern_results[sequence_str] = all_results
                augmentations_created += len(all_results)
                
                print(f"    Successfully created: {len(all_results)}")
            
            # If we need more, create more from the most common patterns
            if augmentations_created < need_total:
                remaining = need_total - augmentations_created
                print(f"\n  Need {remaining} more augmentations, creating from most common patterns...")
                
                # Sort patterns by number of originals (descending)
                sorted_patterns = sorted(label_patterns.items(), 
                                       key=lambda x: len(x[1]), 
                                       reverse=True)
                
                for sequence_str, original_files in sorted_patterns:
                    if remaining <= 0:
                        break
                    
                    stages = sequence_str.split('_')
                    additional = min(remaining, 10)  # Create up to 10 more per pattern
                    
                    # Use random reference files from this pattern
                    for _ in range(min(additional, remaining)):
                        if remaining <= 0:
                            break
                        ref_file = random.choice(original_files)
                        results = self.create_augmented_from_sequence(
                            stages,
                            reference_file=ref_file,
                            num_augmentations=1,
                            min_length=500,
                            target_label=label
                        )
                        
                        if results:
                            pattern_results[sequence_str].extend(results)
                            augmentations_created += len(results)
                            remaining -= len(results)
                    
                    if len(pattern_results[sequence_str]) > 0:
                        print(f"    Created more from pattern: {sequence_str[:50]}...")
            
            # Save all augmented files
            label_path = os.path.join(train_dir, label)
            os.makedirs(label_path, exist_ok=True)
            
            total_saved = 0
            for sequence_str, results in pattern_results.items():
                for combined_df, metadata, reference_file in results:
                    # Generate unique filename
                    filename = f"augmented_seq_{uuid.uuid4().hex}.csv"
                    filepath = os.path.join(label_path, filename)
                    
                    # Save CSV
                    combined_df.to_csv(filepath, index=False)
                    
                    # Save metadata with reference file information
                    metadata_with_ref = {
                        'reference_file': reference_file,
                        'segments': metadata
                    }
                    metadata_path = filepath.replace('.csv', '_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata_with_ref, f, indent=2)
                    
                    total_saved += 1
            
            print(f"\n  Total saved: {total_saved} augmented {label} files")
            print(f"  Final count: {existing_counts[label]} original + {total_saved} augmented = {existing_counts[label] + total_saved} total")


def main():
    """Main function"""
    data_root = "/Users/matteobonsignore/EDM/data/Option 2"
    train_dir = os.path.join(data_root, "Train")
    
    # Target counts (total including originals)
    # User wants: 10 augmented Normal, 154 augmented NPT (220-66), 145 augmented OD (220-75)
    # But we calculate from existing originals:
    target_counts = {
        'Normal': 66,   # 56 existing + 10 augmented = 66 total
        'NPT': 171,     # 17 existing + 154 augmented = 171 total (220 - 66 + 17)
        'OD': 164       # 19 existing + 145 augmented = 164 total (220 - 75 + 19)
    }
    
    # Initialize augmenter
    augmenter = SequencePreservingAugmenter(data_root, seed=42)
    
    # Step 1: Build segment database from ALL training data
    augmenter.build_segment_database(train_dir)
    
    # Step 2: Analyze original timeseries patterns
    augmenter.analyze_original_timeseries(train_dir, list(target_counts.keys()))
    
    # Step 3: Augment dataset
    augmenter.augment_dataset(train_dir, target_counts)
    
    print("\n" + "="*70)
    print("Augmentation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

