import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# --- SCRIPT INSTRUCTIONS ---
#
# 1.  Make sure your data file 'rqa_results_newest_data.csv' is in the same directory.
#
# 2.  Install necessary libraries if you haven't already:
#     pip install seaborn pingouin
#
# 3.  The script is set up to remove outliers > 2.5 SD from the mean. You can change
#     the SD_THRESHOLD variable if you wish.
#
# 4.  To analyze a different RQA measure, change the RQA_MEASURE_TO_ANALYZE variable.
#
# 5.  Run this script from your terminal: python analyze_rqa_results.py

# --- USER CONFIGURATION ---
# The RQA measure you want to analyze from your CSV file
RQA_MEASURE_TO_ANALYZE = 'DET'  # Options: 'RR', 'DET', 'L_avg', 'LAM', 'TT', etc.

# The Standard Deviation threshold for outlier removal
SD_THRESHOLD = 2.5


def run_statistical_analysis(data_filepath, dv_measure, sd_thresh):
    """
    Loads RQA results, removes outliers using the SD method, and performs
    visualization and statistical analysis.
    """
    # --- Load the Data ---
    try:
        df = pd.read_csv(data_filepath)
        print(f"Successfully loaded RQA results from: {data_filepath}")
        if dv_measure not in df.columns:
            print(f"FATAL ERROR: The measure '{dv_measure}' is not a column in your data file.")
            print(f"Available columns are: {df.columns.tolist()}")
            return
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{data_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    print(f"\n--- Analysis started for RQA measure: {dv_measure} ---")


    # --- STEP 1: IDENTIFY AND REMOVE OUTLIERS ---
    print(f"\n[Step 1] Checking for outliers using the {sd_thresh} SD rule...")
    
    original_trial_count = len(df)
    
    def remove_outliers_by_sd(df, group_cols, value_col, threshold):
        """Identifies and removes outliers from a dataframe based on the SD rule."""
        def remove_group_outliers(group):
            mean = group[value_col].mean()
            std_dev = group[value_col].std()
            if pd.isna(std_dev) or std_dev == 0:
                return group
            lower_bound = mean - threshold * std_dev
            upper_bound = mean + threshold * std_dev
            return group[(group[value_col] >= lower_bound) & (group[value_col] <= upper_bound)]
        return df.groupby(group_cols, group_keys=False).apply(remove_group_outliers)

    df_cleaned = remove_outliers_by_sd(df, 
                                       group_cols=['Robot', 'Difficulty', 'ParticipantID'],
                                       value_col=dv_measure,
                                       threshold=sd_thresh)
    
    final_trial_count = len(df_cleaned)
    outliers_removed_count = original_trial_count - final_trial_count
    
    if original_trial_count > 0:
        percentage_lost = (outliers_removed_count / original_trial_count) * 100
        print(f"  Original trial count: {original_trial_count}")
        print(f"  Removed {outliers_removed_count} outlier(s), which is {percentage_lost:.2f}% of the data.")
        print(f"  Final trial count for analysis: {final_trial_count}")
    else:
        print("  No trials to process.")
        
    df = df_cleaned
    print("[Step 1] Outlier removal complete.")


    # --- STEP 2: PREPARE DATA AND VISUALIZE ---
    print("\n[Step 2] Preparing data and generating plots...")

    # Define the mapping from old names to new, descriptive names
    robot_name_map = {
        "Ryan condition": "Joint condition",
        "Ivan condition": "Disjoint condition",
        "Carl condition": "Control condition"
    }
    # Apply the mapping to the 'Robot' column
    df['Robot'] = df['Robot'].map(robot_name_map)
    
    # Define the desired order for the new names
    robot_order = ["Joint condition", "Disjoint condition", "Control condition"]
    
    # Check if all expected robot conditions are present in the data after mapping
    actual_robots = df['Robot'].unique()
    if all(robot in actual_robots for robot in robot_order):
        df['Robot'] = pd.Categorical(df['Robot'], categories=robot_order, ordered=True)
        print(f"  Condition names updated and custom plot order set: {robot_order}")
    else:
        print(f"  Warning: Not all robots in 'robot_order' were found in the data after mapping. Using default alphabetical order.")
        print(f"  Robots in data: {list(actual_robots)}")
    
    sns.set(style="whitegrid", context="talk")

    # Box plot for the main effect of 'Robot'
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Robot', y=dv_measure, data=df, palette="pastel")
    sns.stripplot(x='Robot', y=dv_measure, data=df, color=".25", alpha=0.3)
    plt.title(f'Effect of Robot Condition on {dv_measure} (Outliers Removed)')
    plt.tight_layout()
    plt.show()

    # Box plot for the main effect of 'Difficulty'
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='Difficulty', y=dv_measure, data=df, palette="pastel")
    sns.stripplot(x='Difficulty', y=dv_measure, data=df, color=".25", alpha=0.3)
    plt.title(f'Effect of Difficulty on {dv_measure} (Outliers Removed)')
    plt.tight_layout()
    plt.show()

    # --- MODIFICATION START: Replaced interaction plot with a bar chart ---
    # This bar chart shows the mean value for 'easy' and 'hard' conditions
    # side-by-side for each robot condition.
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Robot', y=dv_measure, hue='Difficulty', data=df,
                palette="colorblind", errorbar='se', capsize=.05)
    plt.title(f'Interaction of Robot and Difficulty on {dv_measure} (Outliers Removed)')
    plt.ylabel(f'Mean {dv_measure}')
    plt.legend(title='Difficulty')
    plt.tight_layout()
    plt.show()
    # --- MODIFICATION END ---
    
    print("[Step 2] Plots generated and displayed.")


    # --- STEP 3: PERFORM THE TWO-WAY REPEATED MEASURES ANOVA ---
    print(f"\n[Step 3] Performing Two-Way Repeated Measures ANOVA for '{dv_measure}'...")
    aov = pg.rm_anova(data=df, dv=dv_measure, within=['Robot', 'Difficulty'],
                      subject='ParticipantID', detailed=True)
    print("\n--- ANOVA Results ---")
    print(aov)


    # --- STEP 4: PERFORM POST-HOC TESTS (IF NECESSARY) ---
    is_robot_significant = aov.loc[aov['Source'] == 'Robot', 'p-unc'].iloc[0] < 0.05
    is_interaction_significant = aov.loc[aov['Source'] == 'Robot * Difficulty', 'p-unc'].iloc[0] < 0.05

    if is_robot_significant or is_interaction_significant:
        print(f"\n[Step 4] ANOVA showed significant effects. Performing post-hoc pairwise tests...")
        posthocs = pg.pairwise_tests(data=df, dv=dv_measure, within=['Robot', 'Difficulty'],
                                     subject='ParticipantID', padjust='bonf')
        print("\n--- Post-Hoc Test Results ---")
        pd.set_option('display.max_rows', None)
        print(posthocs)
    else:
        print("\n[Step 4] No significant effects requiring post-hoc tests were found in the main ANOVA.")

    
    # --- STEP 5: SIMPLE MAIN EFFECTS ANALYSIS ---
    print("\n\n=======================================================================")
    print("[Step 5] Simple Main Effects: Testing Robot effect at each Difficulty Level")
    print("=======================================================================\n")
    
    print("--- Analysis for 'hard' trials only ---")
    hard_df = df[df['Difficulty'] == 'hard'].copy()
    aov_hard = pg.rm_anova(data=hard_df, dv=dv_measure, within='Robot', subject='ParticipantID', detailed=True)
    print("\n--- ANOVA for 'hard' trials only ---")
    print(aov_hard)
    posthocs_hard = pg.pairwise_tests(data=hard_df, dv=dv_measure, within='Robot', subject='ParticipantID', padjust='bonf')
    print("\n--- Post-Hoc Tests for 'hard' trials only ---")
    print(posthocs_hard)

    print("\n\n--- Analysis for 'easy' trials only ---")
    easy_df = df[df['Difficulty'] == 'easy'].copy()
    aov_easy = pg.rm_anova(data=easy_df, dv=dv_measure, within='Robot', subject='ParticipantID', detailed=True)
    print("\n--- ANOVA for 'easy' trials only ---")
    print(aov_easy)
    posthocs_easy = pg.pairwise_tests(data=easy_df, dv=dv_measure, within='Robot', subject='ParticipantID', padjust='bonf')
    print("\n--- Post-Hoc Tests for 'easy' trials only ---")
    print(posthocs_easy)


    print(f"\n--- Analysis for '{dv_measure}' is complete. ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    data_file = "robot_appearance_rqa_results_newest_data.csv"
    
    # Run the entire analysis workflow using the parameters from the top of the script
    run_statistical_analysis(data_filepath=data_file,
                             dv_measure=RQA_MEASURE_TO_ANALYZE,
                             sd_thresh=SD_THRESHOLD)