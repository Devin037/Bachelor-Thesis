import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import os

# --- USER CONFIGURATION ---
FULL_DATA_FILE = "newest_combined_eyetracking_data.csv"
# The directory where results will be saved
OUTPUT_DIR = "analysis_results_fixation_count"

# Outlier threshold
SD_THRESHOLD = 2.5

def run_proportional_fixation_analysis(df, target_aoi_col, target_aoi_name, sd_thresh):
    """
    Calculates Proportional Fixation Count for a given AOI, removes outliers,
    runs a 2x3 repeated-measures ANOVA, and generates visualizations.
    """
    print("\n" + "="*80)
    print(f"Running Analysis for AOI: '{target_aoi_name}' (Column: {target_aoi_col})")
    print("="*80)

    # --- 1. Calculate Proportional Fixation Count ---
    print("\n[Step 1] Calculating Proportional Fixation Count...")
    
    # Isolate only fixation events from the main dataframe
    fixations_df = df[df['Eye movement type'] == 'Fixation'].copy()
    
    # Filter for fixations on the current target AOI
    aoi_fixations = fixations_df[fixations_df[target_aoi_col] == True]
    
    # Group by trial and COUNT the fixations for the AOI
    aoi_fix_counts = aoi_fixations.groupby(['ParticipantID', 'classification_timeframe_number']).size().to_frame(name='AOI_Fixation_Count').reset_index()

    # Calculate TOTAL number of fixations for each trial
    total_fix_counts = fixations_df.groupby(['ParticipantID', 'classification_timeframe_number']).size().to_frame(name='Total_Trial_Fixation_Count').reset_index()

    # Create a complete list of all trials to merge onto
    all_trials = df[['ParticipantID', 'classification_timeframe_number', 'Robot', 'difficulty']].drop_duplicates()
    
    # Merge AOI counts and total trial counts
    analysis_df = pd.merge(all_trials, aoi_fix_counts, on=['ParticipantID', 'classification_timeframe_number'], how='left')
    analysis_df = pd.merge(analysis_df, total_fix_counts, on=['ParticipantID', 'classification_timeframe_number'], how='left')

    # Fill NaNs and calculate the proportion
    analysis_df['AOI_Fixation_Count'].fillna(0, inplace=True)
    analysis_df['Total_Trial_Fixation_Count'].fillna(0, inplace=True) # A trial might have no fixations at all
    analysis_df['Proportional_Fixation_Count'] = np.where(analysis_df['Total_Trial_Fixation_Count'] > 0,
                                                          analysis_df['AOI_Fixation_Count'] / analysis_df['Total_Trial_Fixation_Count'],
                                                          0)
    print(f"Proportional Fixation Count calculated for {len(analysis_df)} trials.")

    # --- 2. Outlier Removal ---
    print(f"\n[Step 2] Checking for outliers in 'Proportional_Fixation_Count' for '{target_aoi_name}'...")
    original_rows = len(analysis_df)
    def remove_outliers_by_sd(df, group_cols, value_col, threshold):
        def remove_group_outliers(group):
            mean = group[value_col].mean()
            std_dev = group[value_col].std()
            if pd.isna(std_dev) or std_dev == 0: return group
            lower_bound = mean - threshold * std_dev
            upper_bound = mean + threshold * std_dev
            return group[(group[value_col] >= lower_bound) & (group[value_col] <= upper_bound)]
        return df.groupby(group_cols, group_keys=False).apply(remove_group_outliers)

    analysis_df = remove_outliers_by_sd(analysis_df,
                                          group_cols=['Robot', 'difficulty', 'ParticipantID'],
                                          value_col='Proportional_Fixation_Count',
                                          threshold=sd_thresh)
    outliers_removed = original_rows - len(analysis_df)
    percentage_lost = (outliers_removed / original_rows) * 100 if original_rows > 0 else 0
    print(f" Removed {outliers_removed} outlier(s) ({percentage_lost:.2f}% of the data).")
    
    # Save the cleaned data to a unique file
    output_filename = f"proportional_fixation_count_{target_aoi_name.replace(' ', '_').lower()}_results.csv"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    analysis_df.to_csv(output_filepath, index=False)
    print(f"Cleaned results for '{target_aoi_name}' saved to '{output_filepath}'.")

    # --- 3. Visualize and Analyze ---
    print(f"\n[Step 3] Visualizing and running ANOVA for '{target_aoi_name}'...")
    analysis_df['Proportional_Fixation_Count_Percent'] = analysis_df['Proportional_Fixation_Count'] * 100.0
    
    robot_name_map = {
        "Ryan condition": "Joint condition",
        "Ivan condition": "Disjoint condition",
        "Carl condition": "Control condition"
    }
    analysis_df['Robot'] = analysis_df['Robot'].map(robot_name_map)
    robot_order = ["Joint condition", "Disjoint condition", "Control condition"]
    if all(robot in analysis_df['Robot'].unique() for robot in robot_order):
        analysis_df['Robot'] = pd.Categorical(analysis_df['Robot'], categories=robot_order, ordered=True)

    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Robot', y='Proportional_Fixation_Count_Percent', hue='difficulty', data=analysis_df, palette="magma", capsize=.05, errorbar="se")
    plt.title(f"Mean Proportional Fixation Count on {target_aoi_name}\nby Condition and Difficulty")
    plt.ylabel('Mean Proportional Fixation Count (%)')
    plt.xlabel('Robotic Condition')
    plt.legend(title='Difficulty')
    # Save the plot to a file
    plot_filename = f"plot_fixation_count_{target_aoi_name.replace(' ', '_').lower()}.png"
    plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(plot_filepath)
    plt.show()
    print(f"Plot for '{target_aoi_name}' saved to '{plot_filepath}'.")

    # Perform ANOVA
    aov = pg.rm_anova(data=analysis_df,
                      dv='Proportional_Fixation_Count', 
                      within=['Robot', 'difficulty'],
                      subject='ParticipantID',
                      detailed=True)
    print(f"\n--- ANOVA Results for Proportional Fixation Count on '{target_aoi_name}' ---")
    pg.print_table(aov)

    # Conditional Post-Hoc tests
    is_robot_sig = aov.loc[aov['Source'] == 'Robot', 'p-unc'].iloc[0] < 0.05
    is_interaction_sig = aov.loc[aov['Source'] == 'Robot * difficulty', 'p-unc'].iloc[0] < 0.05
    if is_robot_sig or is_interaction_sig:
        print(f"\n--- Post-Hoc Tests for Proportional Fixation Count on '{target_aoi_name}' ---")
        posthocs = pg.pairwise_tests(data=analysis_df, dv='Proportional_Fixation_Count', within=['Robot', 'difficulty'], subject='ParticipantID', padjust='bonf')
        print(posthocs)

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. Load and Prepare Data ONCE ---
    print("Loading and preparing main data file...")
    try:
        main_df = pd.read_csv(FULL_DATA_FILE, decimal=',')
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find '{FULL_DATA_FILE}'")
        exit()

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Standard data cleaning
    main_df['classification_timeframe_number'] = pd.to_numeric(main_df['classification_timeframe_number'], errors='coerce')
    main_df.dropna(subset=['classification_timeframe_number'], inplace=True)
    main_df['classification_timeframe_number'] = main_df['classification_timeframe_number'].astype('Int64')
    if 'ParticipantID' not in main_df.columns: main_df['ParticipantID'] = 'Unknown'
    else: main_df['ParticipantID'] = main_df['ParticipantID'].ffill().bfill()
    grouping_cols_for_ffill = ['ParticipantID', 'classification_timeframe_number']
    cols_to_ffill = ['Robot', 'difficulty']
    for col_ffill in cols_to_ffill:
        if col_ffill in main_df.columns:
            main_df[col_ffill] = main_df.groupby(grouping_cols_for_ffill, group_keys=False)[col_ffill].ffill().bfill()
    print("Data loaded and prepared.")

    # --- 2. Create Combined AOI Column ---
    print("\nCreating combined 'Classification Buttons' AOI...")
    main_df['classification_buttons'] = main_df['is_true_category'] | main_df['is_false_category']
    print("Combined AOI created.")

    # --- 3. Define AOIs and Run Analysis for Each ---
    aois_to_analyze = [
        {'col': 'is_face', 'name': 'Robot Face'},
        {'col': 'is_cards', 'name': 'Cards'},
        {'col': 'classification_buttons', 'name': 'Classification Buttons'}
    ]

    for aoi in aois_to_analyze:
        run_proportional_fixation_analysis(df=main_df.copy(),
                                           target_aoi_col=aoi['col'], 
                                           target_aoi_name=aoi['name'], 
                                           sd_thresh=SD_THRESHOLD)

    print("\nAll fixation count analyses complete.")