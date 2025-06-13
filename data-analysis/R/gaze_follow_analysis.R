# -----------------------------------------------------------------------------
# Script: statistics_gaze_follow_analysis.R
# Purpose: Load raw trial-level data (totalgaze.csv), process variables
#          relevant to gaze following behavior, calculate descriptive
#          statistics with plots, and conduct inferential statistics
#          (GLMM and SDT) for gaze following.
# -----------------------------------------------------------------------------

# --- 1. SETUP: Load Necessary Packages ---
library(tidyverse)
library(lme4)      # For GLMM
library(car)       # For Anova function
library(emmeans)   # For post-hoc tests and plotting interactions
library(scales)    # For percent_format
library(afex)      # For repeated-measures ANOVA (for SDT)
library(patchwork) # For combining plots into a single figure

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

# --- 3. STANDARDIZE COLUMN NAMES & INITIAL TRANSFORMATIONS ---
participant_id_original_name <- "participant"
robot_col_original_name <- "Robot"
difficulty_input_col_name <- "difficulty"
correct_side_original_name <- "correct_side"
participants_side_choice_original_name <- "participants_side_choice"
gaze_decision_original_name <- "gazeDecision"

data_gaze_following <- data_raw

participant_id_col <- participant_id_original_name
robot_col <- robot_col_original_name
difficulty_original_col <- difficulty_input_col_name
correct_side_col <- correct_side_original_name
participant_choice_col <- participants_side_choice_original_name
gaze_decision_col <- gaze_decision_original_name

difficulty_labelled_col <- "Difficulty_Condition"

# --- 4. PREPARE FACTORS ---
cat("\n\n--- 4. Preparing IVs as Factors for Gaze Following Analysis ---\n")

if (robot_col %in% colnames(data_gaze_following) && !is.factor(data_gaze_following[[robot_col]])) {
  data_gaze_following[[robot_col]] <- factor(data_gaze_following[[robot_col]], levels = c("Carl condition", "Ivan condition", "Ryan condition"))
  cat(paste0("Converted '", robot_col, "' to factor.\n"))
}
if (participant_id_col %in% colnames(data_gaze_following) && !is.factor(data_gaze_following[[participant_id_col]])) {
  data_gaze_following[[participant_id_col]] <- as.factor(data_gaze_following[[participant_id_col]])
  cat(paste0("Converted '", participant_id_col, "' to factor.\n"))
}

if (difficulty_original_col %in% colnames(data_gaze_following)) {
  data_gaze_following[[difficulty_labelled_col]] <- factor(tolower(data_gaze_following[[difficulty_original_col]]),
                                                           levels = c("easy", "hard"),
                                                           labels = c("Easy", "Hard"))
  cat(paste0("Created '", difficulty_labelled_col, "'.\n"))
} else {
  stop(paste0("Original difficulty column '",difficulty_original_col,"' not found."))
}
cat("--- Gaze following data preparation complete. ---\n")

# --- DATASET FOR MODELS (used in all subsequent analyses) ---
model_data_gaze <- data_gaze_following %>%
  filter(tolower(!!sym(gaze_decision_col)) %in% c("left", "right")) %>%
  mutate(
    robot_gaze_correct_val = ifelse(is.na(!!sym(gaze_decision_col)) | is.na(!!sym(correct_side_col)), NA_integer_,
                                    ifelse(as.character(!!sym(gaze_decision_col)) == as.character(!!sym(correct_side_col)), 1, 0)),
    robot_gaze_correct = factor(robot_gaze_correct_val, levels = c(0,1), labels = c("Incorrect Gaze", "Correct Gaze")),
    gaze_followed_val = ifelse(is.na(!!sym(participant_choice_col)) | is.na(!!sym(gaze_decision_col)), NA_integer_,
                               ifelse(as.character(!!sym(participant_choice_col)) == as.character(!!sym(gaze_decision_col)), 1, 0))
  ) %>%
  filter(!!sym(robot_col) %in% c("Ryan condition", "Ivan condition")) %>%
  filter(!is.na(gaze_followed_val) & !is.na(robot_gaze_correct) &
           !is.na(!!sym(difficulty_labelled_col)) & !is.na(!!sym(participant_id_col))) %>%
  mutate(
    Robot_Condition_Model = droplevels(factor(.data[[robot_col]])),
    Difficulty_Model = factor(.data[[difficulty_labelled_col]]),
    Participant_ID_Model = factor(.data[[participant_id_col]]),
    Gaze_Correctness_Model = factor(robot_gaze_correct)
  )

# --- 5. DESCRIPTIVE STATISTICS & VISUALIZATION ---
cat("\n\n--- 5. Generating Descriptive Statistics and Plots ---\n")

if (exists("model_data_gaze") && nrow(model_data_gaze) > 0) {
  # Calculate counts for each condition (Easy/Hard)
  descriptive_summary <- model_data_gaze %>%
    group_by(Robot_Condition_Model, Gaze_Correctness_Model, Difficulty_Model) %>%
    summarise(
      n_followed = sum(gaze_followed_val, na.rm = TRUE),
      n_total_trials = n(),
      .groups = 'drop'
    )
  
  # --- [NEW] 5.0.1 DISPLAY DESCRIPTIVE PERCENTAGES IN CONSOLE ---
  cat("\n\n--- 5.0.1. Gaze Following Percentages by Condition ---\n")
  
  # Calculate and format percentages for clear console output
  descriptive_percentages <- descriptive_summary %>%
    mutate(
      percentage_followed = (n_followed / n_total_trials),
      # Format for printing
      percentage_str = scales::percent(percentage_followed, accuracy = 0.1),
      # Relabel robot conditions to "Joint" and "Disjoint" for clarity
      Robot_Condition_Display = case_when(
        Robot_Condition_Model == "Ryan condition" ~ "Joint (Ryan)",
        Robot_Condition_Model == "Ivan condition" ~ "Disjoint (Ivan)",
        TRUE ~ as.character(Robot_Condition_Model)
      )
    ) %>%
    # Select and reorder columns for a clean table view
    select(
      Robot_Condition_Display, 
      Difficulty_Model, 
      Gaze_Correctness_Model, 
      percentage_str,
      n_followed,
      n_total_trials
    ) %>%
    # Arrange for easy reading
    arrange(Robot_Condition_Display, Difficulty_Model, Gaze_Correctness_Model)
  
  print(descriptive_percentages, n = Inf) # n = Inf ensures all rows are printed
  # --- [END NEW SECTION] ---
  
  # --- 5.1 Visualization of Descriptive Statistics ---
  cat("\n\n--- 5.1. Creating Bar Charts for Descriptive Gaze Following ---\n")
  
  # Prepare data for plotting by calculating percentages and relabeling
  descriptive_plot_data <- descriptive_summary %>%
    mutate(
      percentage_followed = (n_followed / n_total_trials),
      # Relabel robot conditions to "Joint" and "Disjoint"
      Robot_Condition_Plot = case_when(
        Robot_Condition_Model == "Ryan condition" ~ "Joint",
        Robot_Condition_Model == "Ivan condition" ~ "Disjoint",
        TRUE ~ as.character(Robot_Condition_Model)
      )
    ) %>%
    # Set the order for the factor so the legend and colors are correct
    mutate(Robot_Condition_Plot = factor(Robot_Condition_Plot, levels = c("Joint", "Disjoint")))
  
  # Custom theme to match the python plot style
  theme_custom_style <- function() {
    theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"), # Center title
        panel.border = element_rect(colour = "black", fill=NA, linewidth=1), # Add border
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(linetype = "dashed", color = "grey80"),
        panel.grid.minor.y = element_blank(),
        legend.title = element_blank(), # Remove legend title
        axis.title.x = element_blank() # Remove x-axis title from individual plots
      )
  }
  
  # Plot 1: Gaze Following after CORRECT Gaze Cues
  plot_desc_correct <- descriptive_plot_data %>%
    filter(Gaze_Correctness_Model == "Correct Gaze") %>%
    ggplot(aes(x = Difficulty_Model, y = percentage_followed, fill = Robot_Condition_Plot)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
    scale_y_continuous(labels = scales::percent_format(accuracy=1), limits = c(0, 1.01), expand = c(0, 0)) +
    scale_fill_manual(values = c("Joint" = "skyblue", "Disjoint" = "steelblue")) +
    labs(title = "Correct Gaze Following", y = "Percentage") +
    theme_custom_style()
  
  # Plot 2: Gaze Following after INCORRECT Gaze Cues
  plot_desc_incorrect <- descriptive_plot_data %>%
    filter(Gaze_Correctness_Model == "Incorrect Gaze") %>%
    ggplot(aes(x = Difficulty_Model, y = percentage_followed, fill = Robot_Condition_Plot)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
    scale_y_continuous(labels = scales::percent_format(accuracy=1), limits = c(0, 1.01), expand = c(0, 0)) +
    scale_fill_manual(values = c("Joint" = "lightcoral", "Disjoint" = "indianred")) +
    labs(title = "Incorrect Gaze Following", y = NULL) + # Remove y-axis title for shared axis
    theme_custom_style()
  
  # Combine the plots side-by-side using patchwork
  combined_plot <- plot_desc_correct + plot_desc_incorrect +
    plot_annotation(
      title = 'Gaze Following Behavior (Easy vs Hard)',
      theme = theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
    )
  
  cat("\n--- Displaying Combined Descriptive Plot ---\n")
  print(combined_plot)
  
  
} else {
  cat("\n--- Skipping Descriptive Statistics & Plots: 'model_data_gaze' not available or empty. ---\n")
}

# --- 6. COMPLEMENTARY GLMM ANALYSIS (FOR APPENDIX) ---
cat("\n\n--- 6. Complementary GLMM Analysis (For Appendix) ---\n")

if (nrow(model_data_gaze) > 50 && n_distinct(model_data_gaze$Participant_ID_Model) > 1) {
  options(contrasts = c("contr.sum", "contr.poly"))
  gaze_follow_glmm <- NULL
  tryCatch({
    formula_str <- "gaze_followed_val ~ Robot_Condition_Model * Gaze_Correctness_Model * Difficulty_Model + (1 | Participant_ID_Model)"
    gaze_follow_glmm <- glmer( as.formula(formula_str), data = model_data_gaze,
                               family = binomial(link = "logit"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    cat("--- GLMM fitting successful. ---\n")
    cat("\n--- ANOVA Table (Type III Wald Chi-square tests) for GLMM ---\n")
    print(Anova(gaze_follow_glmm, type = "III"))
  }, error = function(e) { cat("--- ERROR during GLMM fitting: ---\n"); print(e) })
}

# --- 7. PRIMARY ANALYSIS: SIGNAL DETECTION THEORY (SDT) ---
cat("\n\n\n--- 7. Primary Analysis: Signal Detection Theory (SDT) ---\n")

if (exists("model_data_gaze") && nrow(model_data_gaze) > 0) {
  
  # --- 7.1. Calculate SDT Counts ---
  cat("\n--- 7.1. Calculating SDT counts per participant and condition ---\n")
  sdt_counts <- model_data_gaze %>%
    mutate(
      sdt_outcome = case_when(
        gaze_followed_val == 1 & robot_gaze_correct_val == 1 ~ "Hit",
        gaze_followed_val == 0 & robot_gaze_correct_val == 1 ~ "Miss",
        gaze_followed_val == 1 & robot_gaze_correct_val == 0 ~ "False Alarm",
        gaze_followed_val == 0 & robot_gaze_correct_val == 0 ~ "Correct Rejection"
      )
    ) %>%
    group_by(Participant_ID_Model, Robot_Condition_Model, Difficulty_Model) %>%
    summarise(
      n_hits = sum(sdt_outcome == "Hit", na.rm = TRUE),
      n_misses = sum(sdt_outcome == "Miss", na.rm = TRUE),
      n_fas = sum(sdt_outcome == "False Alarm", na.rm = TRUE),
      n_crs = sum(sdt_outcome == "Correct Rejection", na.rm = TRUE),
      .groups = 'drop'
    )
  
  # --- 7.2. Calculate d' and c ---
  cat("\n--- 7.2. Calculating d' (sensitivity) and c (criterion) ---\n")
  sdt_results <- sdt_counts %>%
    mutate(
      # Apply log-linear correction to prevent infinite values
      H = (n_hits + 0.5) / (n_hits + n_misses + 1),
      FA = (n_fas + 0.5) / (n_fas + n_crs + 1),
      d_prime = qnorm(H) - qnorm(FA),
      criterion_c = -0.5 * (qnorm(H) + qnorm(FA))
    )
  
  # --- 7.3. Inferential Statistics on d' and c ---
  cat("\n--- 7.3. Running Repeated Measures ANOVAs on d' and c ---\n")
  
  # Analysis 1: Sensitivity (d').
  cat("\n--- ANOVA on d' (Sensitivity) ---\n")
  anova_d_prime <- aov_ez(
    id = "Participant_ID_Model", dv = "d_prime", data = sdt_results,
    within = c("Robot_Condition_Model", "Difficulty_Model")
  )
  print(summary(anova_d_prime))
  
  # Analysis 2: Bias (c).
  cat("\n--- ANOVA on c (Bias/Criterion) ---\n")
  anova_criterion_c <- aov_ez(
    id = "Participant_ID_Model", dv = "criterion_c", data = sdt_results,
    within = c("Robot_Condition_Model", "Difficulty_Model")
  )
  print(summary(anova_criterion_c))
  
  # --- 7.4. Post-Hoc Analysis for Significant Main Effects ---
  cat("\n--- 7.4. Post-Hoc analysis for significant main effect of Robot on Criterion (c) ---\n")
  emm_c_robot <- emmeans(anova_criterion_c, ~ Robot_Condition_Model)
  print(summary(emm_c_robot))
  
  # --- 7.5. Visualization of SDT Results ---
  cat("\n--- 7.5. Creating Bar Charts for d' and c ---\n")
  
  # Create a summary dataframe with means and CIs for plotting
  sdt_summary_for_plotting <- sdt_results %>%
    group_by(Robot_Condition_Model, Difficulty_Model) %>%
    summarise(
      mean_d_prime = mean(d_prime, na.rm = TRUE),
      se_d_prime = sd(d_prime, na.rm = TRUE) / sqrt(n()),
      ci_d_prime = se_d_prime * qt(0.975, df = n() - 1),
      mean_c = mean(criterion_c, na.rm = TRUE),
      se_c = sd(criterion_c, na.rm = TRUE) / sqrt(n()),
      ci_c = se_c * qt(0.975, df = n() - 1),
      .groups = 'drop'
    ) %>%
    # RENAME AND REORDER the Robot Condition factor for plotting
    mutate(
      Robot_Condition_Model = case_when(
        Robot_Condition_Model == "Ryan condition" ~ "Joint Condition",
        Robot_Condition_Model == "Ivan condition" ~ "Disjoint Condition",
        TRUE ~ as.character(Robot_Condition_Model)
      ),
      Robot_Condition_Model = factor(Robot_Condition_Model, levels = c("Joint Condition", "Disjoint Condition"))
    )
  
  # Plot 1: Sensitivity (d') - Now with updated names and order
  plot_d_prime <- ggplot(sdt_summary_for_plotting,
                         aes(x = Difficulty_Model, y = mean_d_prime, fill = Robot_Condition_Model)) +
    geom_bar(stat = "identity", position = position_dodge(0.9), color = "black", width = 0.8) +
    geom_errorbar(aes(ymin = mean_d_prime - ci_d_prime, ymax = mean_d_prime + ci_d_prime),
                  position = position_dodge(0.9), width = 0.25, linewidth = 0.5) +
    scale_fill_brewer(palette = "Pastel1", name = "Robot Condition") +
    labs(title = "Sensitivity to Gaze Cue Validity",
         subtitle = "Participants' ability to discriminate correct from incorrect gaze cues.",
         x = "Task Difficulty",
         y = "Sensitivity (d')") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold"),
          axis.title = element_text(face = "bold"))
  
  cat("\n--- Displaying Sensitivity (d') Plot ---\n")
  print(plot_d_prime)
  
  # Plot 2: Response Criterion (c) - Now with updated names and order
  plot_criterion_c <- ggplot(sdt_summary_for_plotting,
                             aes(x = Robot_Condition_Model, y = mean_c, fill = Difficulty_Model)) +
    geom_bar(stat = "identity", position = position_dodge(0.9), color = "black", width = 0.8) +
    geom_errorbar(aes(ymin = mean_c - ci_c, ymax = mean_c + ci_c),
                  position = position_dodge(0.9), width = 0.25, linewidth = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey30") +
    scale_fill_brewer(palette = "Pastel2", name = "Task Difficulty") +
    labs(title = "Response Bias for Following Gaze Cues",
         subtitle = "A negative value indicates a liberal bias (tendency to follow).",
         x = "Robot Condition",
         y = "Response Criterion (c)") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold"),
          axis.title = element_text(face = "bold"))
  
  cat("\n--- Displaying Response Criterion (c) Plot ---\n")
  print(plot_criterion_c)
  
} else {
  cat("\n--- Skipping SDT analysis: 'model_data_gaze' not available or empty. ---\n")
}

cat("\n\n--- End of script processing. ---\n")