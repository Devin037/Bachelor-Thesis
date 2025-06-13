import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.image_generator import ImageGenerator
import traceback # For printing detailed error information

# --- USER: Define your parameters here ---

# 1. Name of the column containing the time series data you want to analyze for RQA
TIME_SERIES_COLUMN_FOR_RQA = 'Gaze point X (MCSnorm)'

# 2. RQA Parameters:
embedding_dim = 3
time_del = 10
threshold_radius_type = 'std_fraction'
threshold_value = 0.1

# Minimum number of data points in a trial required to perform RQA
MIN_DATA_POINTS_PER_TRIAL = 50

# --- Helper function to perform RQA ---
def calculate_rqa_for_series(series_data, emb_dim, t_delay, thresh_type, thresh_val):
    """Calculates RQA measures for a given time series."""
    if len(series_data) < MIN_DATA_POINTS_PER_TRIAL:
        print(f"    Skipping RQA: Not enough data points ({len(series_data)} < {MIN_DATA_POINTS_PER_TRIAL})")
        return None, None

    time_series_obj = TimeSeries(series_data, embedding_dimension=emb_dim, time_delay=t_delay)

    current_radius = 0.0
    if thresh_type == 'std_fraction':
        series_std = np.std(series_data)
        if series_std > 0:
            current_radius = thresh_val * series_std
        else:
            print(f"    Warning: Standard deviation is zero. Using a small fixed radius (0.01).")
            current_radius = 0.01
    elif thresh_type == 'fixed':
        current_radius = thresh_val
    else:
        print(f"    Warning: Unknown threshold_radius_type '{thresh_type}'. Defaulting to 'std_fraction'.")
        series_std = np.std(series_data)
        if series_std > 0:
            current_radius = thresh_val * series_std
        else:
            print(f"    Warning: Standard deviation is zero. Using a small fixed radius (0.01).")
            current_radius = 0.01

    if current_radius <= 0:
        print(f"    Warning: Calculated radius is non-positive ({current_radius}). Setting to a small positive value (0.001).")
        current_radius = 0.001

    settings = Settings(time_series_obj,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(current_radius),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    try:
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        
        rqa_measures = {
            'RR': result.recurrence_rate,
            'DET': result.determinism,
            'L_avg': result.average_diagonal_line,
            'L_max': result.longest_diagonal_line,
            'L_entr': result.entropy_diagonal_lines,
            'LAM': result.laminarity,
            'TT': result.trapping_time,
            'V_max': result.longest_vertical_line,
            'RP_threshold': current_radius
        }
        
        rp_matrix = None
        if hasattr(result, 'recurrence_matrix_reverse'):
            rp_matrix = result.recurrence_matrix_reverse
        else:
            print("    Warning: Recurrence matrix not found in result. Plotting will be skipped for this trial.")
            
        return rqa_measures, rp_matrix

    except Exception as e:
        print(f"    An unrecoverable error occurred during RQA computation: {e}")
        return None, None


# --- Main analysis script ---
def main_analysis(csv_filepath, output_rqa_csv_file="rqa_results.csv", example_plot_trial_id=1):
    """
    Main function to load data, preprocess, run RQA per trial, and save results.
    """
    print("Starting eye-tracking RQA analysis...")

    try:
        df = pd.read_csv(csv_filepath, decimal=',', na_values=['NA', ''])
        print(f"CSV data loaded successfully from: {csv_filepath}")
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return
    except Exception as e:
        print(f"Error loading CSV file '{csv_filepath}': {e}")
        return

    print("Performing initial data cleaning and preprocessing...")

    cols_to_convert_numeric = [
        'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
        'Pupil diameter left', 'Pupil diameter right',
        'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)'
    ]
    for col in cols_to_convert_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected numeric column '{col}' not found in CSV.")

    # UPDATED: Changed the required column name here
    required_columns = [TIME_SERIES_COLUMN_FOR_RQA, 'Eyetracker timestamp', 'robot_appearance_timeframe_number']
    for col in required_columns:
        if col not in df.columns:
            print(f"FATAL ERROR: A required column '{col}' is missing from the CSV. Cannot proceed.")
            return

    df.dropna(subset=[TIME_SERIES_COLUMN_FOR_RQA, 'Eyetracker timestamp'], inplace=True)
    print(f"  Rows after dropping essential NaNs (in '{TIME_SERIES_COLUMN_FOR_RQA}' or 'Eyetracker timestamp'): {len(df)}")
    if df.empty:
        print("  DataFrame is empty after dropping essential NaNs. Cannot proceed.")
        return

    aoi_cols = ['is_cards', 'is_eyes', 'is_face', 'is_false_category',
                'is_robot', 'is_robot_name', 'is_true_category']
    for col in aoi_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.lower().map({'true': True, 'false': False, '': False}).fillna(False).astype(bool)
            else:
                df[col] = df[col].fillna(False).astype(bool)
        else:
            print(f"Warning: Expected AOI column '{col}' not found. It will be treated as False.")
            df[col] = False

    df['classification_category'] = (df.get('is_false_category', False) |
                                     df.get('is_true_category', False))
    print("  'classification_category' column created.")

    # UPDATED: Using the new timeframe column
    df['robot_appearance_timeframe_number'] = pd.to_numeric(df['robot_appearance_timeframe_number'], errors='coerce')
    df.dropna(subset=['robot_appearance_timeframe_number'], inplace=True)
    if df.empty:
        print("  DataFrame is empty after dropping NA 'robot_appearance_timeframe_number'. No trials to process.")
        return
    df['robot_appearance_timeframe_number'] = df['robot_appearance_timeframe_number'].astype('Int64')
    print(f"  Rows after dropping NA trial numbers: {len(df)}")

    if 'ParticipantID' not in df.columns:
        print("Warning: 'ParticipantID' column not found. Creating a dummy 'Unknown' ParticipantID.")
        df['ParticipantID'] = 'Unknown'
    else:
        df['ParticipantID'] = df['ParticipantID'].ffill().bfill()

    # UPDATED: Grouping for ffill now uses the new timeframe column
    grouping_cols_for_ffill = ['ParticipantID', 'robot_appearance_timeframe_number']
    cols_to_ffill = ['Robot', 'difficulty']
    for col_ffill in cols_to_ffill:
        if col_ffill in df.columns:
             df[col_ffill] = df.groupby(grouping_cols_for_ffill, group_keys=False)[col_ffill].ffill()
             df[col_ffill] = df.groupby(grouping_cols_for_ffill, group_keys=False)[col_ffill].bfill()
        else:
            print(f"Warning: Column '{col_ffill}' for IV not found. It will not be included in results.")
            df[col_ffill] = 'N/A'

    print(f"  Using '{TIME_SERIES_COLUMN_FOR_RQA}' for RQA time series.")

    # --- 3. Perform RQA Trial-by-Trial ---
    all_rqa_results = []
    print("\nStarting RQA computation per trial...")

    # UPDATED: Main groupby now uses the new timeframe column
    grouped_trials = df.groupby(['ParticipantID', 'robot_appearance_timeframe_number'])

    for (participant_id, trial_num), trial_data in grouped_trials:
        print(f"\n  Processing Participant: {participant_id}, Trial: {trial_num}")

        robot_condition = trial_data['Robot'].iloc[0] if not trial_data['Robot'].empty else 'N/A'
        difficulty_level = trial_data['difficulty'].iloc[0] if not trial_data['difficulty'].empty else 'N/A'

        print(f"    Robot: {robot_condition}, Difficulty: {difficulty_level}")

        time_series_for_rqa = trial_data[TIME_SERIES_COLUMN_FOR_RQA].dropna().values

        rqa_output, rp_matrix = calculate_rqa_for_series(time_series_for_rqa,
                                                       embedding_dim,
                                                       time_del,
                                                       threshold_radius_type,
                                                       threshold_value)

        if rqa_output:
            print(f"    RQA successful for P{participant_id}, Trial {trial_num}.")
            trial_results = {
                'ParticipantID': participant_id,
                'Trial': trial_num,
                'Robot': robot_condition,
                'Difficulty': difficulty_level,
                'NumDataPoints': len(time_series_for_rqa),
                **rqa_output
            }
            all_rqa_results.append(trial_results)

            if trial_num == example_plot_trial_id and rp_matrix is not None:
                plot_filename = f"recurrence_plot_participant_{participant_id}_trial_{trial_num}.png"
                try:
                    ImageGenerator.save_recurrence_plot(rp_matrix, plot_filename)
                    print(f"    Example recurrence plot saved as {plot_filename}")

                    img = plt.imread(plot_filename)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img, cmap='binary', origin='lower')
                    plt.title(f"RP: P{participant_id}, T{trial_num} ({TIME_SERIES_COLUMN_FOR_RQA})\nRobot: {robot_condition}, Diff: {difficulty_level}")
                    plt.xlabel("Time Index")
                    plt.ylabel("Time Index")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"    Could not display/save example recurrence plot: {e}")
        else:
            print(f"    RQA failed or skipped for P{participant_id}, Trial {trial_num}.")

    # --- 4. Save Aggregated RQA Results ---
    if all_rqa_results:
        results_df = pd.DataFrame(all_rqa_results)
        results_df.to_csv(output_rqa_csv_file, index=False, decimal='.')
        print(f"\nAggregated RQA results saved to: {output_rqa_csv_file}")
        print("\n--- First 5 rows of RQA results ---")
        print(results_df.head())
        print("------------------------------------")
    else:
        print("\nNo RQA results were generated. Check data processing steps and trial lengths.")

    print("\nAnalysis complete.")
    print(f"Next steps: Analyze '{output_rqa_csv_file}' with your second script.")


# --- Run the analysis with your actual CSV file ---
if __name__ == "__main__":
    # Define the path to your data file and the name for your output file.
    actual_csv_filepath = "newest_combined_eyetracking_data.csv"
    output_filename = "robot_appearance_rqa_results_newest_data.csv"
    example_trial_to_plot = 1

    print(f"--- Attempting to run analysis with data from: {actual_csv_filepath} ---")

    try:
        main_analysis(csv_filepath=actual_csv_filepath,
                      output_rqa_csv_file=output_filename,
                      example_plot_trial_id=example_trial_to_plot)
    except FileNotFoundError:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: The file '{actual_csv_filepath}' was not found.")
        print(f"Please ensure the file is in the same directory as the Python script,")
        print(f"or provide the full path to the file (e.g., '/path/to/your/{actual_csv_filepath}').")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    except Exception as e:
        print(f"An unexpected error occurred during the analysis: {e}")
        traceback.print_exc()