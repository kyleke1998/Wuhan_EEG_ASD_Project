library(tidyverse)
library(pROC)
library(dplyr)
library(ggplot2)
library(gridExtra)


setwd("C:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/Code")



metadata = read.csv("C:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/eeg_data/wuhan_study_clinical_data.csv")
metadata = metadata %>% filter(((EOEC=='EO') | (EOEC=='EC')) & ((exp_group=='ASD') | (exp_group=='No ASD')))
metadata$EOEC = as.factor(metadata$EOEC)
metadata$sex = as.factor(metadata$sex)
metadata$exp_group = as.factor(metadata$exp_group)
metadata$subject_id = as.integer(metadata$subject_id)
metadata$trt = ifelse(metadata$exp_group == 'No ASD',0,1)



features_eo = read.csv("C:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/features/features_ml_eo.csv")

merged_data = merge(features_eo, metadata, by = "file_name", all =FALSE)
merged_data = select(merged_data, -sex.x,-age_months.x,-trt.x,CARS_Categorical.x) 
merged_data$sex = merged_data$sex.y
merged_data$CARS_Categorical = merged_data$CARS_Categorical.y
merged_data$trt = merged_data$trt.y
merged_data$exp_group = merged_data$exp_group.x

# rank 1 gamma_coh_P4_C3

plot_correlation <- function(col_name) {
  # Subset data based on column name
  sub_data <- merged_data %>% filter(!is.na(gamma_coh_P4_C3))
 
  # Separate data by sex
  male_data <- sub_data %>% filter(sex == "m")
  female_data <- sub_data %>% filter(sex == "f")
  
  # Calculate Spearman's correlation and p-value for male and female data
  male_cor <- cor.test(male_data$CARS_Numerical, male_data$gamma_coh_P4_C3, method = "spearman")
  female_cor <- cor.test(female_data$CARS_Numerical, female_data$gamma_coh_P4_C3, method = "spearman")
  
  # Create ROC curve and best cutoff for male
  roc_data <- male_data %>% select(trt, gamma_coh_P4_C3) %>% 
    roc(response = trt, predictor = gamma_coh_P4_C3)
  
  results <- coords(roc_data, "best",ret=c("threshold",
                                           "specificity", "sensitivity"),
                    best.method="closest.topleft")
  sensitivity_m = results$sensitivity
  best_cutoff_m = results$threshold
  specificity_m = results$specificity

  # Create ROC curve and best cutoff for female
  roc_data <- female_data %>% select(trt, gamma_coh_P4_C3) %>% 
    roc(response = trt, predictor = gamma_coh_P4_C3)
  
  results <- coords(roc_data, "best",ret=c("threshold",
                                           "specificity", "sensitivity"),
                    best.method="closest.topleft")
  sensitivity_f = results$sensitivity
  best_cutoff_f = results$threshold
  specificity_f = results$specificity
  
  
  # Create the two ggplots with titles including Spearman's correlation and p-value
  ggplot_male <- ggplot(male_data, aes(y = CARS_Numerical, x = gamma_coh_P4_C3, color = exp_group)) + 
    geom_point(size = 3) + 
    xlab("gamma_coh_p4_C3") + ylab("CARS Numeric") + 
    scale_color_discrete(name = "Disease Status") + 
    ggtitle(paste0("Male Data (r = ", round(male_cor$estimate, 2), ", p = ", signif(male_cor$p.value, digits = 2), ")")) +
    theme_classic() +
    geom_vline(xintercept = best_cutoff_m, linetype = "dotted", color = "black") +
    annotate("text", x = 0.6, y = max(male_data$gamma_coh_P4_C3) - 1, 
             label = paste0("Sensitivity: ", round(sensitivity_m, 2), "\n", "Specificity: ", round(specificity_m, 2)))
  
  ggplot_female <- ggplot(female_data, aes(y = CARS_Numerical, x = gamma_coh_P4_C3, color = exp_group)) + 
    geom_point(size = 3) + 
    xlab("gamma_coh_p4_C3") + ylab("CARS Numeric") + 
    scale_color_discrete(name = "Disease Status") + 
    ggtitle(paste0("Female Data (r = ", round(female_cor$estimate, 2), ", p = ", signif(female_cor$p.value, digits = 2), ")")) +
    theme_classic() +
    geom_vline(xintercept = best_cutoff_f, linetype = "dotted", color = "black") +
    annotate("text", x = 0.6, y = max(female_data$gamma_coh_P4_C3) - 1, 
             label = paste0("Sensitivity: ", round(sensitivity_f, 2), "\n", "Specificity: ", round(specificity_f, 2)))
  
  # Arrange the two plots side by side and display them
  grid.arrange(ggplot_male, ggplot_female, ncol = 2)
}

plot_correlation()




plot_correlation <- function(col_name) {
  # Convert col_name to a name object and then a character string
  col_name <- as.name(substitute(col_name))
  
  # Subset data based on column name
  sub_data <- merged_data %>% filter(!is.na({{col_name}}))
  
  # Separate data by sex
  male_data <- sub_data %>% filter(sex == "m")
  female_data <- sub_data %>% filter(sex == "f")
  
  # Calculate Spearman's correlation and p-value for male and female data
  male_cor <- cor.test(male_data$CARS_Numerical, male_data[[col_name]], method = "spearman")
  female_cor <- cor.test(female_data$CARS_Numerical, female_data[[col_name]], method = "spearman")
  
  # Create ROC curve and best cutoff for male
  roc_data_m <- roc_(data = male_data,response = "trt", predictor = as.character(col_name))
  results_m <- coords(roc_data_m, "best", ret = c("threshold", "specificity", "sensitivity"), best.method = "closest.topleft")
  sensitivity_m <- results_m$sensitivity
  best_cutoff_m <- results_m$threshold
  specificity_m <- results_m$specificity
  
  # Create ROC curve and best cutoff for female
  roc_data_f <- roc_(data = female_data, response = "trt", predictor = as.character(col_name))
  results_f <- coords(roc_data_f, "best", ret = c("threshold", "specificity", "sensitivity"), best.method = "closest.topleft")
  sensitivity_f <- results_f$sensitivity
  best_cutoff_f <- results_f$threshold
  specificity_f <- results_f$specificity
  
  # Create the two ggplots with titles including Spearman's correlation and p-value
  ggplot_male <- ggplot(male_data, aes(y = CARS_Numerical, x = {{col_name}}, color = exp_group)) + 
    geom_point(size = 3) + 
    xlab(as.character(substitute(col_name))) + ylab("CARS Numeric") + 
    scale_color_discrete(name = "Disease Status") + 
    ggtitle(paste0("Male Data (r = ", round(male_cor$estimate, 2), ", p = ", signif(male_cor$p.value, digits = 2), ")")) +
    theme_classic() +
    geom_vline(xintercept = best_cutoff_m, linetype = "dotted", color = "black") +
    annotate("text", x = 1, y = max(male_data[['CARS_Numerical']],na.rm=T), 
             label = paste0("Sensitivity: ", round(sensitivity_m, 2), "\n", "Specificity: ", round(specificity_m, 2)))
  
  ggplot_female <- ggplot(female_data, aes(y = CARS_Numerical, x = {{col_name}}, color = exp_group)) + 
    geom_point(size = 3) + 
    xlab(as.character(substitute(col_name))) + ylab("CARS Numeric") + 
    scale_color_discrete(name = "Disease Status") + 
    ggtitle(paste0("Male Data (r = ", round(male_cor$estimate, 2), ", p = ", signif(male_cor$p.value, digits = 2), ")")) +
    theme_classic() +
    geom_vline(xintercept = best_cutoff_m, linetype = "dotted", color = "black") +
    annotate("text", x = 1, y = max(female_data[['CARS_Numerical']],na.rm=T), 
             label = paste0("Sensitivity: ", round(sensitivity_m, 2), "\n", "Specificity: ", round(specificity_m, 2)))
  # Arrange the two plots side by side and display them
  grid.arrange(ggplot_male, ggplot_female, ncol = 2)
  
}

plot_correlation('gamma_coh_P4_C3')



plot_correlation <- function(col_name) {
  # Convert col_name to a name object and then a character string
  col_name <- as.name(substitute(col_name))
  
  # Subset data based on column name
  sub_data <- merged_data %>% filter(!is.na({{col_name}}))
  
  # Separate data by sex
  male_data <- sub_data %>% filter(sex == "m")
  female_data <- sub_data %>% filter(sex == "f")
  
  # Calculate Spearman's correlation and p-value for male and female data
  male_cor <- cor.test(male_data$CARS_Numerical, male_data[[col_name]], method = "spearman")
  female_cor <- cor.test(female_data$CARS_Numerical, female_data[[col_name]], method = "spearman")
  
  # Create ROC curve and best cutoff for male
  roc_data_m <- roc_(data = male_data,response = "trt", predictor = as.character(col_name))
  results_m <- coords(roc_data_m, "best", ret = c("threshold", "specificity", "sensitivity"), best.method = "closest.topleft")
  sensitivity_m <- results_m$sensitivity
  best_cutoff_m <- results_m$threshold
  specificity_m <- results_m$specificity
  
  # Create ROC curve and best cutoff for female
  roc_data_f <- roc_(data = female_data, response = "trt", predictor = as.character(col_name))
  results_f <- coords(roc_data_f, "best", ret = c("threshold", "specificity", "sensitivity"), best.method = "closest.topleft")
  sensitivity_f <- results_f$sensitivity
  best_cutoff_f <- results_f$threshold
  specificity_f <- results_f$specificity
  
  # Create the two ggplots with titles including Spearman's correlation and p-value
  ggplot_male <- ggplot(male_data, aes(y = CARS_Numerical, x = {{col_name}}, color = exp_group)) + 
    geom_point(size = 3) + 
    xlab(as.character(substitute(col_name))) + ylab("CARS Numeric") + 
    scale_color_discrete(name = "Disease Status") + 
    ggtitle(paste0("Male Data\n r = ", round(male_cor$estimate, 2), ", p = ", signif(male_cor$p.value, digits = 2), "\nSensitivity: ", round(sensitivity_m, 2), ", Specificity: ", round(specificity_m, 2))) +
    theme_classic() +
    geom_vline(xintercept = best_cutoff_m, linetype = "dotted", color = "black") +
    annotate("text", x = 0.5, y = max(male_data[[col_name]]) - 1, 
             label = paste0("Cutoff: ", round(best_cutoff_m, 2)))
  
  ggplot_female <- ggplot(female_data, aes(y = CARS_Numerical, x = {{col_name}}, color = exp_group)) + 
    geom_point(size = 3) + 
    xlab(as.character(substitute(col_name))) + ylab("CARS Numeric") + 
    scale_color_discrete(name = "Disease Status") + 
    ggtitle(paste0("Female Data\n  r = ", round(female_cor$estimate, 2), ", p = ", signif(female_cor$p.value, digits = 2), "\nSensitivity: ", round(sensitivity_f, 2), ", Specificity: ", round(specificity_f, 2))) +
    theme_classic() +
    geom_vline(xintercept = best_cutoff_f, linetype = "dotted", color = "black") +
    annotate("text", x = 0.5, y = max(female_data[[col_name]]) - 1, 
             label = paste0("Cutoff: ", round(best_cutoff_f, 2)))
 
  # Arrange the two plots side by side and display them
  plots = grid.arrange(ggplot_male, ggplot_female, ncol = 2)
  
  # save to a file
  ggsave(paste0(col_name, "_CARS_Numeric.png"), plot=plots)
  
}




top_features = c("gamma_coh_P4_C3", "gamma_coh_P4_Fz", "beta_coh_P4_C3", "beta_coh_P4_Fz", 
                "alpha_coh_Pz_T5", "gamma_coh_Pz_Fz", "delta_coh_Pz_T5", "beta_coh_Pz_Fz",
                "gamma_coh_P4_F4", "gamma_coh_P4_F3")


plot_correlation(col_name = "gamma_coh_P4_C3")
plot_correlation(col_name = "gamma_coh_P4_Fz")
plot_correlation(col_name = "beta_coh_P4_C3")
plot_correlation(col_name = "beta_coh_P4_Fz")
plot_correlation(col_name = "alpha_coh_Pz_T5")
plot_correlation(col_name = "gamma_coh_Pz_Fz")
plot_correlation(col_name = "delta_coh_Pz_T5")
plot_correlation(col_name = "beta_coh_Pz_Fz")
plot_correlation(col_name = "gamma_coh_P4_F4")
plot_correlation(col_name = "gamma_coh_P4_F3")


sum(is.na(male_data$PDI))


















    

