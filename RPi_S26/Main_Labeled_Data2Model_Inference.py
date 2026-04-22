import pandas as pd
import glob
import random

def combine_and_randomize_blocks(output_file="combined_data.csv", labels_file="labels.csv"):
    """
    Identifies contiguous blocks of PAIN and LIGHT, shuffles the blocks 
    as whole units to preserve time-series integrity, and saves 
    features and labels separately.
    """
    all_blocks = []
    
    # Identify CSV files
    csv_files = [f for f in glob.glob("*.csv") if f not in [output_file, labels_file, "data.csv"]]
    
    if not csv_files:
        print("No CSV files found.")
        return

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'Label' not in df.columns:
                continue

            # 1. Filter for target categories
            filtered = df[df['Label'].isin(['PAIN', 'LIGHT'])].copy()

            # 2. Identify contiguous blocks
            # This creates a unique ID every time the label changes
            filtered['block_id'] = (filtered['Label'] != filtered['Label'].shift()).cumsum()

            # 3. Split into a list of DataFrames (one per block)
            for _, block in filtered.groupby('block_id'):
                # Remove the temporary block_id before storing
                all_blocks.append(block.drop(columns=['block_id']))
                
            print(f"Extracted {filtered['block_id'].nunique()} blocks from {file}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if all_blocks:
        # 4. Shuffle the list of blocks (NOT the rows inside them)
        random.shuffle(all_blocks)
        
        # 5. Merge blocks back into one DataFrame
        shuffled_df = pd.concat(all_blocks, ignore_index=True)
        
        # 6. Separate and Save
        labels = shuffled_df[['Label']]
        features = shuffled_df.drop(columns=['Label'])
        
        features.to_csv(output_file, index=False)
        labels.to_csv(labels_file, index=False)
        
        print(f"\nSuccess:")
        print(f" - {len(all_blocks)} total blocks shuffled.")
        print(f" - Features: {output_file}")
        print(f" - Labels: {labels_file}")
    else:
        print("No valid data blocks found.")

if __name__ == "__main__":
    combine_and_randomize_blocks()