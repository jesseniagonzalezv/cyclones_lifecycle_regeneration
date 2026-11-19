"""
 Splot the dataset into Train, Validation, and Test sets based on temporal criteria to prevent data leakage.
 The script reads a CSV file containing storm data, splits it into three subsets based on the '    
--- Split Shapes ---
Train shape: (6529, 14)
Validation shape: (1587, 14)
Test shape: (1566, 14)
Saving datasets to: /home/jgonzale/adaptations_models/cyclone_lifecycle/data...
"""

from pathlib import Path

import pandas as pd


def split_and_save_storm_data(
    input_csv="/home/jgonzale/TC_diagnostic/csv_results/valid_storms_clean.csv",
    base_output_path=None,
    cutoff_eval=2032,
    cutoff_holdout=2041
):
    """
    Loads storm data and performs a temporal split into Train, Validation, and Test sets
    to prevent data leakage, then saves them inside the main project's 'data' folder.
    
    Temporal breakdown:
    - Train (< 2032): ~42 years (approx. 70%) to learn long-term historical patterns.
    - Validation (2032 – 2040): ~9 years (approx. 15%) for hyperparameter tuning.
    - Test (>= 2041): ~9 years (approx. 15%) as a completely untouched holdout.
    """
    
    # If no output path is provided, automatically set it to main_folder/data/
    if base_output_path is None:
        # __file__ is main_folder/preprocessing/split.py
        # .parent -> main_folder/preprocessing/
        # .parent.parent -> main_folder/
        script_dir = Path(__file__).resolve()
        main_folder = script_dir.parent.parent
        base_output_path = main_folder / "data"

    # Ensure the output directory exists (creates it if it doesn't)
    base_output_path = Path(base_output_path)
    base_output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the dataset
    print(f"Loading data from: {input_csv}...")
    valid_storms_df = pd.read_csv(input_csv)
    

    # 2. Split datasets chronologically to prevent data leakage
    
    # Train: Before 2032
    train_df = valid_storms_df[valid_storms_df["season"] < cutoff_eval]

    # Validation: Between 2032 and 2040
    val_df = valid_storms_df[
        (valid_storms_df["season"] >= cutoff_eval) & 
        (valid_storms_df["season"] < cutoff_holdout)
    ]

    # Test: From 2041 onwards
    test_df = valid_storms_df[valid_storms_df["season"] >= cutoff_holdout]

    # 3. Print shapes for verification
    print("--- Split Shapes ---")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 4. Save splits to destination paths using Pathlib
    print(f"Saving datasets to: {base_output_path}...")
    train_df.to_csv(base_output_path / "train.csv", index=False)
    val_df.to_csv(base_output_path / "val.csv", index=False)
    test_df.to_csv(base_output_path / "test.csv", index=False)
    print("Process completed successfully!")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # When executing the script directly, it automatically saves to main_folder/data/
    split_and_save_storm_data()