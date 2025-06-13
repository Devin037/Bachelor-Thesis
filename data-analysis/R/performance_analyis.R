# -----------------------------------------------------------------------------
# Script: statistics_performance_analysis.R
# Purpose: Load raw trial-level data (totalgaze.csv), process variables
#          for task performance (score, move duration), calculate extensive
#          descriptive statistics, conduct outlier checks for move duration.
#          Aggregate key performance DVs per participant per condition,
#          check ANOVA assumptions, perform 3x2 repeated measures ANOVA,
#          and visualize final results.
#
# UPDATED: This script now filters move_duration outliers based on a
#          2.5 SD rule per participant, analyzes accuracy as a percentage,
#          and generates a final bar chart for accuracy results.
#
# UPDATED AGAIN: Robot conditions renamed and reordered. Plots are now grouped
#           by difficulty within each robot condition.
# -----------------------------------------------------------------------------

# --- 1. SETUP: Load Necessary Packages ---
# install.packages(c("tidyverse", "patchwork", "scales", "rstatix", "ggpubr", "emmeans"))

library(tidyverse)
library(patchwork)
library(scales)
library(rstatix)
library(ggpubr)
library(emmeans)

# --- 2. LOAD DATA ---
file_path <- "totalgaze.csv"
data_raw <- NULL

cat(paste0("--- Attempting to load '", file_path, "' ---\n"))
tryCatch({
  data_raw <- read_csv(file_path)
  cat(paste0("--- Successfully loaded '", file_path, "'. ---\n"))
}, error = function(e) {
  cat(paste0("--- ERROR: Could not load '", file_path, "'. ---\n"))
  cat("Error message: ", e$message, "\n")
})

if (is.null(data_raw)) {
  stop("Script cannot proceed because data_raw was not loaded.")
}

cat("\n--- Initial Data Inspection (First few rows of raw data) ---\n"); print(head(data_raw))
cat("\n--- Initial Structure of the raw data (str) ---\n"); str(data_raw)

# --- 3. STANDARDIZE COLUMN NAMES & INITIAL TRANSFORMATIONS ---
participant_id_original_name <- "participant"
robot_col_original_name <- "Robot"
difficulty_input_col_name <- "difficulty"
correct_side_original_name <- "correct_side"
participants_side_choice_original_name <- "participants_side_choice"
move_duration_original_name <- "move_duration"

data <- data_raw

participant_id_col <- participant_id_original_name
robot_col <- robot_col_original_name
difficulty_original_col <- difficulty_input_col_name
correct_side_col <- correct_side_original_name
participant_choice_col <- participants_side_choice_original_name
move_duration_col <- move_duration_original_name

score_col <- "task_score"
difficulty_labelled_col <- "Difficulty_Condition"

if (difficulty_original_col %in% colnames(data)) {
  data <- data %>%
    mutate(
      !!sym(difficulty_original_col) := case_when(
        tolower(.data[[difficulty_original_col]]) == "easy" ~ 0,
        tolower(.data[[difficulty_original_col]]) == "hard" ~ 1,
        TRUE ~ NA_real_
      )
    )
  data[[difficulty_original_col]] <- as.numeric(data[[difficulty_original_col]])
}

if (correct_side_col %in% colnames(data) && participant_choice_col %in% colnames(data)) {
  data <- data %>%
    mutate(
      !!sym(score_col) := ifelse(
        is.na(!!sym(correct_side_col)) | is.na(!!sym(participant_choice_col)),
        NA_integer_,
        ifelse(as.character(!!sym(correct_side_col)) == as.character(!!sym(participant_choice_col)), 1, 0)
      )
    )
  data[[score_col]] <- as.integer(data[[score_col]])
}

# --- 4. PREPARE FACTORS & VERIFY COLUMN TYPES ---
cat("\n\n--- 4. Preparing IVs as Factors & Ensuring DV Numeric Types for Performance Analysis ---\n")

if (robot_col %in% colnames(data) && !is.factor(data[[robot_col]])) {
  
  # <<< CHANGED: Renaming and reordering the robot conditions >>>
  # 1. First, rename the existing values to the new desired names.
  data <- data %>%
    mutate(!!sym(robot_col) := recode(!!sym(robot_col),
                                      "Ryan condition" = "Ryan (Joint)",
                                      "Ivan condition" = "Ivan (Disjoint)",
                                      "Carl condition" = "Carl (Control)"))
  
  # 2. Then, create the factor with the new names in the desired order.
  data[[robot_col]] <- factor(data[[robot_col]], levels = c("Ryan (Joint)", "Ivan (Disjoint)", "Carl (Control)"))
  
  cat("--- Robot conditions have been renamed and reordered. New order: Ryan (Joint), Ivan (Disjoint), Carl (Control) ---\n")
}

if (participant_id_col %in% colnames(data) && !is.factor(data[[participant_id_col]])) {
  data[[participant_id_col]] <- as.factor(data[[participant_id_col]])
}

if (difficulty_original_col %in% colnames(data) && is.numeric(data[[difficulty_original_col]])) {
  data[[difficulty_labelled_col]] <- factor(data[[difficulty_original_col]], levels = c(0, 1), labels = c("Easy", "Hard"))
} else if (difficulty_input_col_name %in% colnames(data) && is.character(data[[difficulty_input_col_name]])) {
  data[[difficulty_labelled_col]] <- factor(tolower(data[[difficulty_input_col_name]]), levels = c("easy", "hard"), labels = c("Easy", "Hard"))
} else {
  stop(paste0("No usable difficulty column found to create the factor '",difficulty_labelled_col,"'."))
}


performance_dvs_to_ensure_numeric <- c(score_col, move_duration_col)
for (dv_check in performance_dvs_to_ensure_numeric) {
  if (dv_check %in% colnames(data)) {
    if (!is.numeric(data[[dv_check]])) {
      data[[dv_check]] <- suppressWarnings(as.numeric(as.character(data[[dv_check]])))
    }
  }
}
cat("--- Performance data preparation complete. ---\n")

# --- 5. TRIAL-LEVEL OUTLIER HANDLING (for 'move_duration') ---
cat("\n\n--- 5. Trial-Level Outlier Visualization and Filtering for '", move_duration_col, "' ---")
if (move_duration_col %in% colnames(data) && is.numeric(data[[move_duration_col]])) {
  
  # --- 5.1 VISUALIZATION (using a Bar Chart of Means for Move Duration BEFORE Filtering) ---
  cat(paste0("\n--- Visualizing Mean '", move_duration_col, "' with SD Error Bars (Trial-Level, BEFORE Filtering) ---\n"))
  if (robot_col %in% colnames(data) && difficulty_labelled_col %in% colnames(data) ) {
    
    # Calculate summary stats for plotting move_duration
    summary_for_duration_plot <- data %>%
      filter(!is.na(!!sym(move_duration_col))) %>%
      group_by(!!sym(robot_col), !!sym(difficulty_labelled_col)) %>%
      summarise(
        Mean_Duration = mean(!!sym(move_duration_col), na.rm = TRUE),
        SD_Duration = sd(!!sym(move_duration_col), na.rm = TRUE),
        .groups = 'drop'
      )
    
    # <<< CHANGED: Plot structure updated to group by difficulty >>>
    md_by_condition_plot <- ggplot(summary_for_duration_plot,
                                   aes(x = !!sym(robot_col), y = Mean_Duration, fill = !!sym(difficulty_labelled_col))) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
      geom_errorbar(aes(ymin = Mean_Duration - SD_Duration, ymax = Mean_Duration + SD_Duration),
                    width = 0.25, position = position_dodge(width = 0.9)) +
      scale_fill_brewer(palette = "Pastel1") +
      labs(title = paste("Mean", move_duration_col, "by Condition (Before Outlier Filtering)"),
           subtitle = "Error bars represent +/- 1 Standard Deviation",
           y = paste("Mean", move_duration_col, "(seconds)"),
           x = "Robot Condition",
           fill = "Difficulty") +
      theme_minimal() +
      theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
    print(md_by_condition_plot)
  }
  
  # --- 5.2 FILTERING (2.5 SD Rule per Participant for move_duration) ---
  cat(paste0("\n--- Filtering '", move_duration_col, "' outliers based on 2.5 SD rule per participant ---\n"))
  initial_rows <- nrow(data)
  cat(paste0("Initial number of trials: ", initial_rows, "\n"))
  
  data <- data %>%
    group_by(!!sym(participant_id_col)) %>%
    mutate(
      mean_dur = mean(!!sym(move_duration_col), na.rm = TRUE),
      sd_dur = sd(!!sym(move_duration_col), na.rm = TRUE),
      upper_bound = mean_dur + (2.5 * sd_dur),
      lower_bound = mean_dur - (2.5 * sd_dur)
    ) %>%
    filter(
      is.na(!!sym(move_duration_col)) | (!!sym(move_duration_col) >= lower_bound & !!sym(move_duration_col) <= upper_bound)
    ) %>%
    ungroup() %>%
    select(-mean_dur, -sd_dur, -upper_bound, -lower_bound) # Clean up helper columns
  
  final_rows <- nrow(data)
  rows_removed <- initial_rows - final_rows
  percent_removed <- (rows_removed / initial_rows) * 100
  
  cat(paste0("Filtered number of trials: ", final_rows, "\n"))
  cat(paste0("Removed ", rows_removed, " trials (", round(percent_removed, 2), "%) as outliers from '", move_duration_col, "'.\n"))
  
  # --- 5.3 VISUALIZATION (using a Bar Chart of Means for Move Duration AFTER Filtering) ---
  cat(paste0("\n--- Visualizing Mean '", move_duration_col, "' with SE Error Bars (Trial-Level, AFTER Filtering) ---\n"))
  if (robot_col %in% colnames(data) && difficulty_labelled_col %in% colnames(data) ) {
    
    # Calculate summary stats for plotting move_duration from the CLEANED data
    summary_for_duration_plot_after <- data %>%
      filter(!is.na(!!sym(move_duration_col))) %>%
      group_by(!!sym(robot_col), !!sym(difficulty_labelled_col)) %>%
      summarise(
        Mean_Duration = mean(!!sym(move_duration_col), na.rm = TRUE),
        # Using Standard Error for the final plot is often better for inference
        SE_Duration = sd(!!sym(move_duration_col), na.rm = TRUE) / sqrt(n()),
        .groups = 'drop'
      )
    
    # <<< CHANGED: Plot structure updated to group by difficulty >>>
    md_by_condition_plot_after <- ggplot(summary_for_duration_plot_after,
                                         aes(x = !!sym(robot_col), y = Mean_Duration, fill = !!sym(difficulty_labelled_col))) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
      # Error bars now represent +/- 1 Standard Error
      geom_errorbar(aes(ymin = Mean_Duration - SE_Duration, ymax = Mean_Duration + SE_Duration),
                    width = 0.25, position = position_dodge(width = 0.9)) +
      scale_fill_brewer(palette = "Pastel1") +
      labs(title = paste("Mean", move_duration_col, "by Condition (After Outlier Filtering)"),
           subtitle = "Error bars represent +/- 1 Standard Error",
           y = paste("Mean", move_duration_col, "(seconds)"),
           x = "Robot Condition",
           fill = "Difficulty") +
      theme_minimal() +
      theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(md_by_condition_plot_after)
  }
  
} else {cat(paste0("\nNote: '", move_duration_col, "' column not found/specified or not numeric. Outlier handling for move_duration skipped.\n"))}


# --- 6. DESCRIPTIVE STATISTICS (Trial-Level DVs on FILTERED data) ---
cat("\n\n--- 6. Descriptive Statistics (Trial-Level Performance DVs on Filtered Data) ---\n")
# 6.1 For 'task_score'
if (score_col %in% colnames(data) && is.numeric(data[[score_col]])) {
  cat(paste0("\n--- 6.1.1 Descriptive Statistics for '", score_col, "' by Robot x Difficulty ---\n"))
  descriptive_stats_score_crossed <- data %>% group_by(!!sym(robot_col), !!sym(difficulty_labelled_col)) %>%
    summarise(N_trials = n(), Mean_Score_Prop = mean(!!sym(score_col), na.rm = TRUE), SD_Score = sd(!!sym(score_col), na.rm = TRUE), .groups = 'drop')
  print(descriptive_stats_score_crossed)
}
# 6.2 For 'move_duration'
if (move_duration_col %in% colnames(data) && is.numeric(data[[move_duration_col]])) {
  cat(paste0("\n--- 6.2.1 Descriptive Statistics for '", move_duration_col, "' by Robot x Difficulty ---\n"))
  descriptive_stats_duration_crossed <- data %>% group_by(!!sym(robot_col), !!sym(difficulty_labelled_col)) %>%
    summarise(N_trials = n(), Mean_Duration = mean(!!sym(move_duration_col), na.rm = TRUE), SD_Duration = sd(!!sym(move_duration_col), na.rm = TRUE), .groups = 'drop')
  print(descriptive_stats_duration_crossed)
}

# --- 7. AGGREGATE PERFORMANCE DATA FOR ANOVA ---
cat("\n\n--- 7. Aggregating Performance Data per Participant for ANOVA ---\n")

data_agg_performance <- NULL # Initialize

if (nrow(data) > 0) {
  data_agg_performance <- data %>%
    group_by(!!sym(participant_id_col), !!sym(robot_col), !!sym(difficulty_labelled_col)) %>%
    summarise(
      Mean_Accuracy_Percent = if(score_col %in% colnames(.)) mean(!!sym(score_col), na.rm = TRUE) * 100 else NA_real_,
      Mean_move_duration = if(move_duration_col %in% colnames(.)) mean(!!sym(move_duration_col), na.rm = TRUE) else NA_real_,
      N_Trials_Per_Condition = n(),
      .groups = 'drop'
    )
  
  cat("\n--- Aggregated Performance DVs for ANOVA (First few rows): ---\n")
  print(head(data_agg_performance))
  cat("\nStructure of aggregated Performance DVs for ANOVA:\n")
  str(data_agg_performance)
} else {
  stop("Error: No data remains after filtering. ANOVA cannot proceed.")
}


# --- 8. ANOVA DATA PREPARATION ---
cat("\n\n--- 8. Preparing Aggregated Data for ANOVA ---\n")

dv_accuracy_anova <- "Mean_Accuracy_Percent"
dv_duration_anova <- "Mean_move_duration"

if (!participant_id_col %in% colnames(data_agg_performance)) stop("Participant ID column missing in aggregated data.")
if (!robot_col %in% colnames(data_agg_performance)) stop("Robot column missing in aggregated data.")
if (!difficulty_labelled_col %in% colnames(data_agg_performance)) stop("Difficulty column missing in aggregated data.")

if (!dv_accuracy_anova %in% colnames(data_agg_performance)) {
  warning(paste0("ANOVA DV '", dv_accuracy_anova, "' not found. Accuracy analyses skipped."))
  dv_accuracy_anova <- NULL
}
if (!dv_duration_anova %in% colnames(data_agg_performance)) {
  warning(paste0("ANOVA DV '", dv_duration_anova, "' not found. Duration analyses skipped."))
  dv_duration_anova <- NULL
}

# --- 9. ASSUMPTION CHECKING (Normality per cell) ---
check_normality_per_cell_anova <- function(df, dv_name, group1_name, group2_name) {
  if (is.null(dv_name) || !dv_name %in% colnames(df)) {
    cat(paste0("\nSkipping normality check: DV '", dv_name, "' not available.\n"))
    return()
  }
  cat(paste0("\n--- Normality Check for ANOVA DV: ", dv_name, " (within each ", group1_name, " x ", group2_name, " cell) ---\n"))
  
  # <<< CHANGED: Updated facetting to match new plot style (group by robot) >>>
  hist_plot <- ggplot(df, aes(x = .data[[dv_name]])) +
    geom_histogram(aes(y = after_stat(density)), bins=10, fill = "skyblue", color = "black", alpha = 0.7, na.rm = TRUE) +
    geom_density(alpha = .2, fill = "#FF6666", na.rm = TRUE) +
    facet_grid(as.formula(paste0("`", group2_name, "` ~ `", group1_name, "`")), scales = "free_y") +
    labs(title = paste("Histograms of", dv_name, "(Aggregated)"), x = dv_name, y = "Density") + theme_minimal()
  print(hist_plot)
  
  # <<< CHANGED: Updated facetting to match new plot style (group by robot) >>>
  qq_plot <- ggpubr::ggqqplot(df, x = dv_name, conf.int = TRUE, ggtheme = theme_minimal(), title = paste("Q-Q Plots of", dv_name, "(Aggregated)")) +
    facet_grid(as.formula(paste0("`", group2_name, "` ~ `", group1_name, "`")), scales = "free")
  print(qq_plot)
  
  normality_tests <- df %>%
    group_by(!!sym(group1_name), !!sym(group2_name)) %>%
    filter(sum(!is.na(.data[[dv_name]])) >= 3) %>%
    summarise( shapiro_w = ifelse(sum(!is.na(.data[[dv_name]])) >=3, shapiro.test(.data[[dv_name]])$statistic, NA_real_),
               shapiro_p = ifelse(sum(!is.na(.data[[dv_name]])) >=3, shapiro.test(.data[[dv_name]])$p.value, NA_real_),
               n_for_test = sum(!is.na(.data[[dv_name]])), .groups = 'drop')
  cat("\n  Shapiro-Wilk Test Results (p > 0.05 suggests normality):\n"); print(normality_tests)
}

if (!is.null(dv_accuracy_anova)) { check_normality_per_cell_anova(data_agg_performance, dv_accuracy_anova, robot_col, difficulty_labelled_col) }
if (!is.null(dv_duration_anova)) { check_normality_per_cell_anova(data_agg_performance, dv_duration_anova, robot_col, difficulty_labelled_col) }

# --- 10. SIGNIFICANCE TESTING: 3x2 REPEATED MEASURES ANOVA ---
perform_rm_anova_integrated <- function(df, dv_col, wid_col, within_factors_cols) {
  if (is.null(dv_col) || !dv_col %in% colnames(df)) {
    cat(paste0("\nSkipping ANOVA: DV '", dv_col, "' not available.\n"))
    return(NULL)
  }
  cat(paste0("\n\n--- Repeated Measures ANOVA for: ", dv_col, " ---\n"))
  
  if(!is.numeric(df[[dv_col]])) {
    cat(paste0("  Warning: DV '", dv_col, "' is not numeric. Attempting conversion.\n"))
    df[[dv_col]] <- suppressWarnings(as.numeric(as.character(df[[dv_col]])))
    if(all(is.na(df[[dv_col]]))) {
      cat(paste0("  ERROR: DV '", dv_col, "' could not be converted to numeric or is all NA. Skipping ANOVA.\n"))
      return(NULL)
    }
  }
  
  n_within_levels <- df %>% select(all_of(within_factors_cols)) %>% n_distinct()
  
  complete_cases_df <- df %>%
    filter(!is.na(.data[[dv_col]])) %>%
    group_by(!!sym(wid_col)) %>%
    filter(n() == n_within_levels) %>%
    ungroup()
  
  n_complete_subjects <- length(unique(complete_cases_df[[wid_col]]))
  
  if(n_complete_subjects < 2) {
    cat(paste0("  Warning: Not enough subjects (found ", n_complete_subjects, ") with complete data for '", dv_col, "' across all conditions. Skipping ANOVA.\n"))
    return(NULL)
  }
  
  cat(paste0("  Performing ANOVA on ", n_complete_subjects, " participants with complete data for ", dv_col, ".\n"))
  
  res_aov_obj <- NULL
  tryCatch({
    res_aov_obj <- anova_test(
      data = complete_cases_df,
      dv = !!sym(dv_col),
      wid = !!sym(wid_col),
      within = within_factors_cols
    )
    cat(paste0("\n  --- ANOVA Results for ", dv_col, " ---\n"))
    print(res_aov_obj)
    
    anova_table <- NULL
    if (is.list(res_aov_obj) && "ANOVA" %in% names(res_aov_obj)) {
      anova_table <- res_aov_obj$ANOVA
    } else if (is.data.frame(res_aov_obj) || is_tibble(res_aov_obj)) {
      anova_table <- res_aov_obj
    } else {
      cat("  Warning: Could not identify the ANOVA table within the anova_test result object.\n")
      return(res_aov_obj)
    }
    
    cat("\n  Key P-values and GES from ANOVA table:\n"); print(anova_table %>% filter(Effect != "(Intercept)") %>% select(Effect, p, ges))
    
    interaction_term_pattern <- paste(within_factors_cols, collapse=":")
    interaction_effect_row <- anova_table %>% filter(Effect == interaction_term_pattern)
    
    if (nrow(interaction_effect_row) == 1 && interaction_effect_row$p < 0.05) {
      cat(paste0("\n  --- Interaction effect '", interaction_term_pattern, "' for '", dv_col, "' was significant (p = ", format(interaction_effect_row$p, digits=3),"). Probing simple effects... ---\n"))
      
      cat(paste0("  Simple main effect of ", within_factors_cols[1], " at each level of ", within_factors_cols[2], ":\n"))
      simple_effects_1 <- complete_cases_df %>%
        group_by(!!sym(within_factors_cols[2])) %>%
        anova_test(formula = as.formula(paste0("`", dv_col, "` ~ `", within_factors_cols[1], "`")),
                   wid = !!sym(wid_col), within = !!sym(within_factors_cols[1])) %>%
        get_anova_table() %>%
        adjust_pvalue(method = "bonferroni")
      print(simple_effects_1)
      
      cat(paste0("\n  Simple main effect of ", within_factors_cols[2], " at each level of ", within_factors_cols[1], ":\n"))
      simple_effects_2 <- complete_cases_df %>%
        group_by(!!sym(within_factors_cols[1])) %>%
        anova_test(formula = as.formula(paste0("`", dv_col, "` ~ `", within_factors_cols[2], "`")),
                   wid = !!sym(wid_col), within = !!sym(within_factors_cols[2])) %>%
        get_anova_table() %>%
        adjust_pvalue(method = "bonferroni")
      print(simple_effects_2)
      
      cat("\n  Consider further pairwise comparisons for significant simple effects with >2 levels using emmeans or pairwise_t_test.\n")
      
    } else {
      cat(paste0("\n  --- Interaction effect '", interaction_term_pattern, "' for '", dv_col, "' was NOT significant or not found. Checking main effects... ---\n"))
      
      main_effect_1_row <- anova_table %>% filter(Effect == within_factors_cols[1]) # Robot
      if (nrow(main_effect_1_row) == 1 && main_effect_1_row$p < 0.05) {
        cat(paste0("\n  --- Main effect of '", within_factors_cols[1], "' for '", dv_col, "' was significant (p = ", format(main_effect_1_row$p, digits=3), "). Pairwise comparisons (Bonferroni)... ---\n"))
        pwc_1 <- complete_cases_df %>%
          pairwise_t_test(as.formula(paste0("`", dv_col, "` ~ `", within_factors_cols[1], "`")),
                          paired = TRUE, p.adjust.method = "bonferroni")
        print(pwc_1)
      } else if (nrow(main_effect_1_row) == 1) {
        cat(paste0("\n  --- Main effect of '", within_factors_cols[1], "' for '", dv_col, "' was NOT significant (p = ", format(main_effect_1_row$p, digits=3), "). ---\n"))
      }
      
      main_effect_2_row <- anova_table %>% filter(Effect == within_factors_cols[2]) # Difficulty
      if (nrow(main_effect_2_row) == 1 && main_effect_2_row$p < 0.05) {
        cat(paste0("\n  --- Main effect of '", within_factors_cols[2], "' for '", dv_col, "' was significant (p = ", format(main_effect_2_row$p, digits=3), "). ---\n"))
        pwc_2 <- complete_cases_df %>%
          pairwise_t_test(as.formula(paste0("`", dv_col, "` ~ `", within_factors_cols[2], "`")),
                          paired = TRUE, p.adjust.method = "bonferroni")
        print(pwc_2)
      } else if (nrow(main_effect_2_row) == 1) {
        cat(paste0("\n  --- Main effect of '", within_factors_cols[2], "' for '", dv_col, "' was NOT significant (p = ", format(main_effect_2_row$p, digits=3), "). ---\n"))
      }
    }
    return(res_aov_obj)
    
  }, error = function(e) {
    cat(paste0("  --- ERROR during Repeated Measures ANOVA for '", dv_col, "': ", e$message, " ---\n"))
    return(NULL)
  })
}

# Perform ANOVA for Mean Accuracy
if (!is.null(data_agg_performance) && !is.null(dv_accuracy_anova)) {
  results_accuracy_anova <- perform_rm_anova_integrated(data_agg_performance, dv_accuracy_anova, participant_id_col, c(robot_col, difficulty_labelled_col))
}

# Perform ANOVA for Mean Duration
if (!is.null(data_agg_performance) && !is.null(dv_duration_anova)) {
  results_duration_anova <- perform_rm_anova_integrated(data_agg_performance, dv_duration_anova, participant_id_col, c(robot_col, difficulty_labelled_col))
}

cat("\n\n--- Performance Analysis Script (with ANOVA) Finished ---\n")

# --- 11. VISUALIZE AGGREGATED ACCURACY RESULTS ---
cat("\n\n--- 11. Visualizing Aggregated Accuracy Performance ---\n")
if (!is.null(data_agg_performance) && dv_accuracy_anova %in% colnames(data_agg_performance)) {
  
  # Calculate summary statistics for the accuracy plot (Mean and Standard Error)
  accuracy_summary_for_plot <- data_agg_performance %>%
    group_by(!!sym(robot_col), !!sym(difficulty_labelled_col)) %>%
    summarise(
      Mean_Accuracy = mean(!!sym(dv_accuracy_anova), na.rm = TRUE),
      SE_Accuracy = sd(!!sym(dv_accuracy_anova), na.rm = TRUE) / sqrt(n()),
      .groups = 'drop'
    )
  cat("\nSummary statistics for accuracy plot:\n")
  print(accuracy_summary_for_plot)
  
  # <<< NOTE: This plot already had the correct structure and will update automatically with the new names/order >>>
  # Create the bar chart for Mean Accuracy Percentage
  accuracy_plot <- ggplot(accuracy_summary_for_plot,
                          aes(x = !!sym(robot_col), y = Mean_Accuracy, fill = !!sym(difficulty_labelled_col))) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    geom_errorbar(aes(ymin = Mean_Accuracy - SE_Accuracy, ymax = Mean_Accuracy + SE_Accuracy),
                  width = 0.25, position = position_dodge(width = 0.9)) +
    scale_fill_brewer(palette = "Pastel1") + # Using a color-blind friendly palette
    labs(title = "Mean Task Accuracy by Robot and Difficulty",
         x = "Robot Condition",
         y = "Mean Accuracy (%)",
         fill = "Difficulty") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "top",
          axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5),
          panel.grid.major.x = element_blank(), # Cleaner look
          panel.grid.minor.y = element_blank()) +
    coord_cartesian(ylim = c(0, 100)) # Ensure Y axis goes from 0 to 100
  
  print(accuracy_plot)
  cat("\n--- Accuracy bar chart generated. ---\n")
  
} else {
  cat("\n--- Skipping accuracy bar chart: Aggregated data or accuracy DV not available. ---\n")
}

cat("\n--- Full Analysis Script Finished ---\n")
cat("Review all ANOVA tables, Mauchly's test results, post-hoc tests, and generated plots carefully.\n")
