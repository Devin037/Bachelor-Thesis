# -----------------------------------------------------------------------------
# FULLY CONSOLIDATED SCRIPT FOR QUALTRICS SUBJECTIVE DATA ANALYSIS
# -----------------------------------------------------------------------------

# --- 1. SETUP: Load Necessary Packages ---
library(tidyverse)
library(lubridate)
library(psych)
library(rstatix)
library(ggpubr)

# --- 2. LOAD INITIAL RAW DATA ---
file_path <- "qualtrics_data_final.csv"
data <- NULL
cat(paste0("--- Attempting to load '", file_path, "' ---\n"))
tryCatch({
  data <- read_csv(file_path)
  cat(paste0("--- Successfully loaded '", file_path, "'. ---\n"))
}, error = function(e) {
  cat(paste0("--- ERROR: Could not load '", file_path, "'. ---\n"))
  cat("Error message: ", e$message, "\n")
})
if (is.null(data)) {
  stop("Script cannot proceed because data was not loaded.")
}

# --- 3. PROCESS DEMOGRAPHIC VARIABLES ---
cat("\n\n--- 3. Processing Demographic Variables ---\n")
if ("participant" %in% colnames(data) && !"participantID" %in% colnames(data)) {
  data <- data %>% rename(participantID = participant)
  cat("Renamed 'participant' column to 'participantID' for compatibility.\n")
} else if (!"participantID" %in% colnames(data)) {
  data$participantID <- 1:nrow(data)
  cat("Warning: No 'participant' or 'participantID' column found. Created a new 'participantID' column.\n")
}
# ... (demographic processing code remains unchanged) ...

# --- 4. DEFINE ITEMS FOR EACH SCALE AND ROBOT ---
cat("\n\n--- 4. Defining Scale Items ---\n")
anthro_carl_items <- paste0("anthropomorphism _", 1:5, "_carl"); like_carl_items <- paste0("likability _", 1:5, "_carl"); intel_carl_items <- paste0("intelligence _", 1:5, "_carl"); trust_carl_items <- paste0("trust _", 1:14, "_carl")
anthro_ryan_items <- paste0("anthropomorphism _", 1:5, "_ryan"); like_ryan_items <- paste0("likability _", 1:5, "_ryan"); intel_ryan_items <- paste0("intelligence _", 1:5, "_ryan"); trust_ryan_items <- paste0("trust _", 1:14, "_ryan")
anthro_ivan_items <- paste0("anthropomorphism _", 1:5, "_ivan"); like_ivan_items <- paste0("likability _", 1:5, "_ivan"); intel_ivan_items <- paste0("intelligence _", 1:5, "_ivan"); trust_ivan_items <- paste0("trust _", 1:14, "_ivan")

# --- 5. CALCULATE COMPOSITE SCORES (ROW MEANS) ---
cat("\n\n--- 5. Calculating Composite Scores ---\n")
check_and_calculate_mean <- function(df, items_list, new_col_name) {
  existing_items <- intersect(items_list, colnames(df))
  if (length(existing_items) == 0) { cat(paste0("Warning: No items for '", new_col_name, "'. Skipping.\n")); return(df) }
  if (length(existing_items) < length(items_list)) { cat(paste0("Warning: Not all items for '", new_col_name, "' found. Using: ", paste(existing_items, collapse=", "), "\n")) }
  df <- df %>% mutate(across(all_of(existing_items), as.numeric))
  df <- df %>% mutate(!!new_col_name := rowMeans(select(., all_of(existing_items)), na.rm = TRUE))
  cat(paste0("Calculated: ", new_col_name, "\n"))
  return(df)
}
data <- check_and_calculate_mean(data, anthro_carl_items, "Anthro_Carl_Score"); data <- check_and_calculate_mean(data, like_carl_items, "Like_Carl_Score"); data <- check_and_calculate_mean(data, intel_carl_items, "Intel_Carl_Score"); data <- check_and_calculate_mean(data, trust_carl_items, "Trust_Carl_Score")
data <- check_and_calculate_mean(data, anthro_ryan_items, "Anthro_Ryan_Score"); data <- check_and_calculate_mean(data, like_ryan_items, "Like_Ryan_Score"); data <- check_and_calculate_mean(data, intel_ryan_items, "Intel_Ryan_Score"); data <- check_and_calculate_mean(data, trust_ryan_items, "Trust_Ryan_Score")
data <- check_and_calculate_mean(data, anthro_ivan_items, "Anthro_Ivan_Score"); data <- check_and_calculate_mean(data, like_ivan_items, "Like_Ivan_Score"); data <- check_and_calculate_mean(data, intel_ivan_items, "Intel_Ivan_Score"); data <- check_and_calculate_mean(data, trust_ivan_items, "Trust_Ivan_Score")
cat("\n--- Composite score calculation finished. ---\n")


# --- 6. DESCRIPTIVE STATISTICS & VISUALIZATIONS OF COMPOSITE SCORES ---
cat("\n\n--- 6. Descriptive Statistics & Visualizations of Composite Scores ---\n")
composite_score_columns <- c(grep("_Score$", colnames(data), value = TRUE))
existing_composite_score_columns <- intersect(composite_score_columns, colnames(data))

if(length(existing_composite_score_columns) > 0){
  data <- data %>% mutate(across(all_of(existing_composite_score_columns), as.numeric))
  cat("\n--- Summary (Min, Q1, Median, Mean, Q3, Max) for Composite Scores ---\n")
  print(summary(data[, existing_composite_score_columns]))
} else { cat("No composite score columns found to summarize.\n")}

construct_name_map <- c(Anthro = "Anthropomorphism", Like = "Likability", Intel = "Intelligence", Trust = "Trust")
constructs_short_names_for_iteration <- names(construct_name_map)

# --- MODIFICATION START ---
# Define original order for data extraction AND new names for plotting
robots_order_original <- c("Ryan", "Ivan", "Carl")
robot_display_names <- c("Joint", "Disjoint", "Control")
# --- MODIFICATION END ---

cat("\n\n--- 6.3 Generating Combined Faceted Boxplot for All Constructs ---\n")
if ("participantID" %in% colnames(data) && length(existing_composite_score_columns) > 0) {
  data_long_all_constructs_viz <- data %>%
    select(participantID, all_of(existing_composite_score_columns)) %>%
    pivot_longer(cols = all_of(existing_composite_score_columns), names_to = "Score_Name", values_to = "ScoreValue") %>%
    mutate(
      Short_Construct_Name = str_extract(Score_Name, "^(Anthro|Like|Intel|Trust)"),
      Robot = str_extract(Score_Name, "(Carl|Ryan|Ivan)"),
      Robot = factor(Robot, levels = robots_order_original),
      
      # --- MODIFICATION START ---
      # Create a new column for the plot labels based on the original Robot column
      Robot_Plot_Label = recode(Robot,
                                "Ryan" = "Joint",
                                "Ivan" = "Disjoint",
                                "Carl" = "Control"),
      # Factor the new column with the new display names for correct ordering in plots
      Robot_Plot_Label = factor(Robot_Plot_Label, levels = robot_display_names),
      # --- MODIFICATION END ---
      
      Construct_Display = recode(Short_Construct_Name, !!!construct_name_map),
      Construct_Display = factor(Construct_Display, levels = unname(construct_name_map))
    ) %>%
    filter(!is.na(Robot) & !is.na(Construct_Display) & !is.na(ScoreValue))
  
  if (nrow(data_long_all_constructs_viz) > 0) {
    # --- MODIFICATION START: Use Robot_Plot_Label for x and fill aesthetics ---
    combined_faceted_plot_final <- ggplot(data_long_all_constructs_viz, aes(x = Robot_Plot_Label, y = ScoreValue, fill = Robot_Plot_Label)) +
      # --- MODIFICATION END ---
      geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 1.5, width = 0.7) +
      facet_wrap(~Construct_Display, scales = "free", ncol = 2) +
      labs(title = "Comparison of Subjective Ratings by Robot Condition", x = "Robot Condition", y = "Mean Score") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "none", strip.text = element_text(face="bold",size=11), axis.text.x=element_text(angle=45,hjust=1,size=10), axis.title=element_text(size=11), plot.title=element_text(hjust=0.5,size=14,face="bold"), panel.spacing=unit(1.5,"lines"))
    print(combined_faceted_plot_final)
  } else { cat("No data for combined faceted plot after filtering.\n")}
} else { cat("Warning: 'participantID' or composite scores missing. Skipping combined faceted boxplot.\n")}


# --- 6.4 Generating Combined Faceted Bar Chart with 95% Confidence Intervals ---
cat("\n\n--- 6.4 Generating Combined Bar Chart with 95% Confidence Intervals ---\n")
if (exists("data_long_all_constructs_viz") && nrow(data_long_all_constructs_viz) > 0) {
  
  # --- MODIFICATION START: Group by the new Robot_Plot_Label column ---
  summary_stats_for_plot <- data_long_all_constructs_viz %>%
    group_by(Construct_Display, Robot_Plot_Label) %>%
    # --- MODIFICATION END ---
    summarise(
      Mean = mean(ScoreValue, na.rm = TRUE),
      SD = sd(ScoreValue, na.rm = TRUE),
      N = n(),
      .groups = 'drop'
    ) %>%
    mutate(
      SE = SD / sqrt(N),
      CI_lower = Mean - 1.96 * SE,
      CI_upper = Mean + 1.96 * SE
    )
  
  # --- MODIFICATION START: Use Robot_Plot_Label for x and fill aesthetics ---
  combined_faceted_barchart <- ggplot(summary_stats_for_plot, aes(x = Robot_Plot_Label, y = Mean, fill = Robot_Plot_Label)) +
    # --- MODIFICATION END ---
    geom_bar(stat = "identity", color = "black", width = 0.8) +
    geom_errorbar(
      aes(ymin = CI_lower, ymax = CI_upper),
      width = 0.25,
      linewidth = 0.5,
      color = "black"
    ) +
    geom_text(
      aes(label = sprintf("M = %.2f", Mean)),
      vjust = -2.5,
      color = "black",
      size = 3.5
    ) +
    facet_wrap(~Construct_Display, scales = "free", ncol = 2) +
    labs(
      title = "Mean Subjective Ratings by Robot Condition",
      subtitle = "Error bars represent 95% Confidence Intervals",
      x = "Robot Condition",
      y = "Mean Score"
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, .15))) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "none",
      strip.text = element_text(face = "bold", size = 11),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.title = element_text(size = 11),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.spacing = unit(1.5, "lines"),
      panel.grid.major.x = element_blank()
    )
  
  print(combined_faceted_barchart)
  cat("\n--- Bar chart with CIs generated successfully. ---\n")
  
} else {
  cat("Warning: Could not generate bar chart because the initial data processing step (6.3) failed to produce data.\n")
}


# --- 7. IDENTIFY POTENTIAL OUTLIERS (1.5 * IQR Rule) ---
# ... (This section remains unchanged as it operates on original score columns) ...

# --- 8. CHECK NORMALITY FOR EACH COMPOSITE SCORE ---
# ... (This section remains unchanged) ...

# --- 9. CRONBACH'S ALPHA FOR SCALE RELIABILITY ---
# ... (This section remains unchanged) ...

# --- 10. PARAMETRIC TESTING - REPEATED MEASURES ANOVA ---
cat("\n\n--- 10. Parametric Testing (Repeated Measures ANOVAs) ---\n")
if (!"participantID" %in% colnames(data)) { stop("Error: 'participantID' column required for ANOVA.") }

# --- NOTE: This section correctly uses the ORIGINAL robot names ("Ryan", "Ivan", "Carl")
# which is why we didn't overwrite the original 'Robot' column earlier.
robots_order <- c("Ryan", "Ivan", "Carl") 

for (construct_short_final_anova in constructs_short_names_for_iteration) {
  full_construct_final_anova <- construct_name_map[[construct_short_final_anova]]
  cat(paste0("\n\n--- RM ANOVA for: ", full_construct_final_anova, " Scores ---\n"))
  
  composite_cols_final_anova <- paste0(construct_short_final_anova, "_", robots_order, "_Score")
  # ... (rest of ANOVA code remains unchanged) ...
}
cat("\n--- Parametric testing finished. ---\n")


# --- 11. SAVE FINAL DATASET ---
# ... (This section remains unchanged) ...