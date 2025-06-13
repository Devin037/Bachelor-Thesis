import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# --- USER CONFIGURATION ---
FULL_DATA_FILE = "newest_combined_eyetracking_data.csv"

# Define your key AOIs and give them short names for the matrix
AOI_DEFINITIONS = {
    'Robot': 'is_robot',
    'Cards': 'is_cards',
    'Classification': 'classification_category'
}

def run_full_transition_analysis(data_filepath, aoi_defs):
    """
    Calculates effective AOI transitions, runs statistical tests (Chi-Squared)
    with post-hoc analysis, and then visualizes the results.
    """
    # --- 1. Load and Prepare Data ---
    print("[Step 1] Loading data and calculating all transitions...")
    try:
        df = pd.read_csv(data_filepath, decimal=',')
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find the data file. {e}")
        return

    # Standard data cleaning
    df.dropna(subset=['robot_appearance_timeframe_number', 'Robot', 'difficulty'], inplace=True)
    df['classification_category'] = (df.get('is_false_category', False) | df.get('is_true_category', False))

    def get_aoi_state(row, aoi_definitions):
        for aoi_name, col_name in aoi_definitions.items():
            if col_name in row and row[col_name]:
                return aoi_name
        return 'Outside'
    df['aoi_state'] = df.apply(lambda row: get_aoi_state(row, aoi_defs), axis=1)

    # --- 2. Build a Master List of ALL Transitions ---
    all_transitions = []
    for name, group in df.groupby(['ParticipantID', 'robot_appearance_timeframe_number']):
        robot_condition = group['Robot'].iloc[0]
        difficulty_level = group['difficulty'].iloc[0]

        simplified_sequence = group['aoi_state'][group['aoi_state'].shift() != group['aoi_state']]
        effective_sequence = simplified_sequence[simplified_sequence != 'Outside']

        if len(effective_sequence) > 1:
            trial_transitions = list(zip(effective_sequence, effective_sequence.iloc[1:]))
            for trans_from, trans_to in trial_transitions:
                all_transitions.append({
                    'From': trans_from,
                    'To': trans_to,
                    'Robot': robot_condition,
                    'Difficulty': difficulty_level
                })

    if not all_transitions:
        print("No transitions were found.")
        return

    master_transition_df = pd.DataFrame(all_transitions)
    print("Master list of all transitions created successfully.")

    # --- 3. Perform Overall Statistical Tests (Chi-Squared) ---
    print("\n[Step 3] Performing Chi-Squared tests for overall significance...")

    # Test 1: Does the transition pattern depend on the Robot?
    print("\n--- Test 1: Do transition patterns differ by ROBOT? ---")
    # The crosstab function creates the contingency table of observed counts
    contingency_table_robot = pd.crosstab(master_transition_df['From'], [master_transition_df['To'], master_transition_df['Robot']])
    chi2, p, dof, expected_robot = chi2_contingency(contingency_table_robot)
    print(f"Chi-Squared Statistic: {chi2:.2f}, p-value: {p:.4f}")
    if p < 0.05:
        print("Conclusion: YES, the pattern of transitions is significantly different across the robot conditions.")
        # --- MODIFICATION START: POST-HOC FOR ROBOT CONDITION ---
        print("\n--- Post-Hoc Analysis: Standardized Residuals for Robot Condition ---")
        print("This shows which specific transitions occurred significantly more or less often than expected for each robot.")
        # Rule of thumb: A residual > 1.96 or < -1.96 is significant at p < .05
        residuals_robot = (contingency_table_robot - expected_robot) / np.sqrt(expected_robot)
        
        # Flatten the table for easier parsing
        stacked_residuals_robot = residuals_robot.stack(level=[0, 1]).reset_index()
        stacked_residuals_robot.columns = ['From', 'To', 'Robot', 'Residual']
        
        # Filter for significant results
        significant_residuals_robot = stacked_residuals_robot[np.abs(stacked_residuals_robot['Residual']) > 1.96]
        
        for index, row in significant_residuals_robot.sort_values(by='Residual', ascending=False).iterrows():
            direction = "more" if row['Residual'] > 0 else "less"
            print(f"  - In '{row['Robot']}', transitions from '{row['From']}' to '{row['To']}' occurred {direction} frequently than expected (Residual: {row['Residual']:.2f})")
        # --- MODIFICATION END ---
    else:
        print("Conclusion: NO, the pattern of transitions is not significantly different across the robot conditions.")


    # Test 2: Does the transition pattern depend on Difficulty?
    print("\n--- Test 2: Do transition patterns differ by DIFFICULTY? ---")
    contingency_table_difficulty = pd.crosstab(master_transition_df['From'], [master_transition_df['To'], master_transition_df['Difficulty']])
    chi2, p, dof, expected_difficulty = chi2_contingency(contingency_table_difficulty)
    print(f"Chi-Squared Statistic: {chi2:.2f}, p-value: {p:.4f}")
    if p < 0.05:
        print("Conclusion: YES, the pattern of transitions is significantly different between easy and hard trials.")
        # --- MODIFICATION START: POST-HOC FOR DIFFICULTY ---
        print("\n--- Post-Hoc Analysis: Standardized Residuals for Difficulty ---")
        print("This shows which specific transitions occurred significantly more or less often than expected for each difficulty level.")
        
        residuals_difficulty = (contingency_table_difficulty - expected_difficulty) / np.sqrt(expected_difficulty)
        
        # Flatten the table for easier parsing
        stacked_residuals_difficulty = residuals_difficulty.stack(level=[0, 1]).reset_index()
        stacked_residuals_difficulty.columns = ['From', 'To', 'Difficulty', 'Residual']
        
        # Filter for significant results
        significant_residuals_difficulty = stacked_residuals_difficulty[np.abs(stacked_residuals_difficulty['Residual']) > 1.96]

        for index, row in significant_residuals_difficulty.sort_values(by='Residual', ascending=False).iterrows():
            direction = "more" if row['Residual'] > 0 else "less"
            print(f"  - In '{row['Difficulty']}' trials, transitions from '{row['From']}' to '{row['To']}' occurred {direction} frequently than expected (Residual: {row['Residual']:.2f})")
        # --- MODIFICATION END ---
    else:
        print("Conclusion: NO, the pattern of transitions is not significantly different between easy and hard trials.")


    # --- 4. Generate Descriptive Heatmaps for Each Condition ---
    print("\n[Step 4] Generating descriptive probability matrices and heatmaps for each condition...")
    
    robot_name_map = {
        "Ryan condition": "Joint Condition (Ryan)",
        "Ivan condition": "Disjoint Condition (Ivan)",
        "Carl condition": "Control Condition (Carl)"
    }
    
    # Loop through each condition to generate its specific matrix and heatmap
    for difficulty in ['easy', 'hard']:
        print("\n" + "#"*30 + f"\n#   ANALYSIS FOR {difficulty.upper()} TRIALS   #\n" + "#"*30 + "\n")
        for robot in ["Ryan condition", "Ivan condition", "Carl condition"]:
            print("\n" + "="*80)
            print(f"CONDITION: {robot} / {difficulty}")
            print("="*80)
            
            condition_subset_df = master_transition_df[
                (master_transition_df['Robot'] == robot) &
                (master_transition_df['Difficulty'] == difficulty)
            ]
            
            if condition_subset_df.empty:
                print("No transitions found for this specific condition.")
                continue

            count_matrix = pd.crosstab(condition_subset_df['From'], condition_subset_df['To'])
            prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).fillna(0)
            
            print("\n--- Effective AOI Transition PROBABILITY Matrix ---")
            print(prob_matrix.to_string(float_format="%.2f"))
            
            descriptive_name = robot_name_map.get(robot, robot)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(prob_matrix, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, vmin=0, vmax=1)
            plt.title(f"AOI Transition Probabilities for {descriptive_name} in '{difficulty}' statements")
            
            plt.xlabel("To AOI")
            plt.ylabel("From AOI")
            plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    run_full_transition_analysis(data_filepath=FULL_DATA_FILE,
                                 aoi_defs=AOI_DEFINITIONS)