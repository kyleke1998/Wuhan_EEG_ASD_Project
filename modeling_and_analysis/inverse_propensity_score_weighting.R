library(twang)
library(tidyverse)
library(broom)
library(survey)


setwd("C:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/Code")

metadata = read.csv("C:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/eeg_data/wuhan_study_clinical_data.csv")
metadata = metadata %>% filter(((EOEC=='EO') | (EOEC=='EC')) & ((exp_group=='ASD') | (exp_group=='No ASD')))
metadata$EOEC = as.factor(metadata$EOEC)
metadata$sex = as.factor(metadata$sex)
metadata$exp_group = as.factor(metadata$exp_group)
metadata$subject_id = as.integer(metadata$subject_id)
#metadata$trt = ifelse(metadata$CARS_Categorical == 'No ASD',0,1)
metadata$trt = ifelse(metadata$exp_group == 'No ASD',0,1)
length_df = read.csv("./length_df.csv")
subjects_too_short = length_df %>% filter(length_remaining < 1) %>% select(subject) %>% pull()





# Loading in feature set of spectral power features as proof of concept     ###### 
all_features = read.csv("../features/all_features.csv")
proof_of_concept_joined = metadata %>% select(c(sex,age_months,file_name,trt,EOEC)) %>%
  inner_join(y=all_features,by='file_name',keep=F)

# turning alpha presence into binary
proof_of_concept_joined$alpha_presence = ifelse(proof_of_concept_joined$alpha_presence == 'True',1,0)
rm(all_features)



proof_of_concept_joined = proof_of_concept_joined %>% filter(!file_name %in% subjects_too_short)

proof_of_concept_joined = proof_of_concept_joined %>% filter(EOEC == 'EO')
  
na_balance = proof_of_concept_joined %>% group_by(trt) %>% 
  summarize(across(everything(), ~mean(is.na(.))))

rm(metadata_iq_filtered)


write.csv(proof_of_concept_joined,"features_ml_eo.csv")












# assumption: missingness don't depend on treatment or covariates
# Overall propensity model
ps.logit <- glm(trt ~ age_months + sex,
                data = proof_of_concept_joined,family='binomial')



proof_of_concept_joined$ps <- predict(ps.logit,type='response') 

proof_of_concept_joined$weights <- ifelse(proof_of_concept_joined$trt == 1, 1/proof_of_concept_joined$ps,
                                         1/(1-proof_of_concept_joined$ps))

proof_of_concept_joined$denominator <- ifelse(proof_of_concept_joined$trt == 1, proof_of_concept_joined$ps,
 (1-proof_of_concept_joined$ps))
proof_of_concept_joined$numerator <-  ifelse(proof_of_concept_joined$trt == 1, mean(proof_of_concept_joined$trt == 1),
                                             mean(proof_of_concept_joined$trt == 0))

proof_of_concept_joined$sw_weights <- proof_of_concept_joined$numerator / proof_of_concept_joined$denominator


proof_of_concept_joined$weights <- proof_of_concept_joined$sw_weights



bal.logit <- dx.wts(x = proof_of_concept_joined$ps,
                    data=proof_of_concept_joined,
                    vars=c("age_months","sex"),
                    treat.var="trt",
                    x.as.weights=F,
                    perm.test.iters=0, estimand = "ATE")

bal.table(bal.logit)




##############################################################

# Building the outcome model


###############################################################


library(betareg)

ipw <- function(phenotype="bp_delta_Fp1", exposure="trt", covariates=c("sex", "age_months","EOEC"), data=matched.data, coh_mod_choice = 'logit') {
  # phenotype: a string that is a phenotype name
  # exposure: a string this is a exposure name 
  # covariates: an array of covariate variables
  print(sum(is.null(data$subclass)))
  
  # create outcome model: phenotype ~ trt + sex + age_months + EOEC
  covariate_string <- paste(covariates, collapse="+")
  if (phenotype != "alpha_presence") {
    
    # check if outcome is spectral coherence
    if (grepl("coh", phenotype) | grepl("bp", phenotype)) {
      
      # chose to model using beta regression
      if (coh_mod_choice == 'beta'){
        print('beta')
        mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
        mod <- betareg(as.formula(mod_string), data = data, weights = data$sw_weights)
      }
      
      # chose to model using logit transformation
      if (coh_mod_choice == 'logit') {
        print('logit')
        mod_string <- sprintf('log(%s/(1-%s)) ~ %s + %s', phenotype, phenotype, exposure, covariate_string)
        mod <- lm(as.formula(mod_string), data = data,weights = data$sw_weights)
      } 
    
    
    }
    
    
    else if (grepl("sample", phenotype) | grepl("sd_", phenotype) | grepl("kurt_", phenotype)){
      print('log outcome')
      mod_string <- sprintf('log(%s) ~ %s + %s', phenotype, exposure, covariate_string)
      mod <- lm(as.formula(mod_string),data = data,weights = data$sw_weights)
      
    }
    
    # if other features that is not spectral coherence, relative power, or sample entropy
    else {
    print('regular regression')
    mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
    mod <- lm(as.formula(mod_string),data = data,weights = data$sw_weights)
    }
    
  # the above formats formula as a string 
   
  }
  
  
  else{
    print('Logistic regression')
    mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
    # the above formats formula as a string 
    mod <- glm(as.formula(mod_string), data = data, family = "binomial",weights = data$sw_weights)
  }
  return(mod)
}



########################### SVYGLM###########################
library(survey)

ipw_svyglm <- function(phenotype="bp_delta_Fp1", exposure="trt", covariates=c("sex", "age_months"), data=matched.data, coh_mod_choice = 'logit') {
  # phenotype: a string that is a phenotype name
  # exposure: a string this is a exposure name 
  # covariates: an array of covariate variables
  mod_design <- svydesign(id =~ 1, weights =~ weights, data = data)
  
  
  # create outcome model: phenotype ~ trt + sex + age_months + EOEC
  covariate_string <- paste(covariates, collapse="+")
  if (phenotype != "alpha_presence") {
    
    # check if outcome is spectral coherence
    if (grepl("coh", phenotype) | grepl("bp", phenotype)) {
      
      # chose to model using beta regression
      if (coh_mod_choice == 'beta'){
        print('beta')
        mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
        mod <- svyglm(as.formula(mod_string), design = mod_design, family = betabinomial())
      }
      
      # chose to model using logit transformation
      if (coh_mod_choice == 'logit') {
        print('logit')
        mod_string <- sprintf('log(%s/(1-%s)) ~ %s + %s', phenotype, phenotype, exposure, covariate_string)
        mod <- svyglm(as.formula(mod_string), design = mod_design, family = gaussian())
      } 
      
      
    }
    
    
    else if (grepl("sample", phenotype) | grepl("sd_", phenotype) | grepl("kurt_", phenotype)){
      print('log outcome')
      mod_string <- sprintf('log(%s) ~ %s + %s', phenotype, exposure, covariate_string)
      mod <- svyglm(as.formula(mod_string), design = mod_design, family = gaussian())
      
    }
    
    # if other features that is not spectral coherence, relative power, or sample entropy
    else {
      print('regular regression')
      mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
      mod <- svyglm(as.formula(mod_string), design = mod_design, family = gaussian())
    }
    
    # the above formats formula as a string 
    
  }
  
  
  else{
    print('Logistic regression')
    mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
    # the above formats formula as a string 
    mod <- svyglm(as.formula(mod_string), design = mod_design, family = binomial())
  }
  return(mod)
}




#############################################################






# feature list
feature_list = colnames(proof_of_concept_joined[, -which(names(proof_of_concept_joined) %in% 
                                                           c("subject_x","subject_y","sex","age_months","EOEC","file_name","trt","index_x","index_y","exp_group","weights","ps","exposure","CARS_Categorical","index","subject",
                                                             'numerator','denominator','sw_weights','subclass','distance'))])


# Create empty list that holds all models
list_of_fits <- list()




## Running through all phenotypes

# loop through all the outcomes in feature list
for (phenotype in unique(feature_list)) {
  assoc_model <- ipw_svyglm(phenotype, exposure="trt", data=matched.data, coh_mod_choice = 'logit')
  summary_tibble <- tidy(assoc_model)
  summary_tibble$outcome <- phenotype
  print(phenotype)
  sample_sizes <- proof_of_concept_joined %>% 
    filter(!is.na(.data[[phenotype]])) %>% 
    count(trt)
  
  summary_tibble$n_ASD <- sample_sizes$n[which(sample_sizes$trt == 1)]
  summary_tibble$n_control <- sample_sizes$n[which(sample_sizes$trt == 0)]
  
  
  list_of_fits <- c(list_of_fits,list(summary_tibble))
}


all_fits = bind_rows(list_of_fits) %>% filter(term== 'trt')



bonferroni_pvalues <- p.adjust(all_fits$p.value, method = "bonferroni")

all_fits$bonf_p.value <- bonferroni_pvalues

fdr_adjusted_pvalues <- p.adjust(all_fits$p.value, method = "fdr")

all_fits$q.value <- fdr_adjusted_pvalues




hist(all_fits$p.value)
top_fits_exp_group_logit_ipw_svyglm = all_fits %>% arrange(p.value) %>% head(50)
# sort the data frame by the q-value column in ascending order
top_fits_exp_group_logit_full_match_svyglm <- all_fits %>% arrange(p.value) %>% head(50)

# Print top_fits to a CSV file named "top_fits.csv" in the working directory
write.csv(top_fits_exp_group_logit_ipw_svyglm, file = "top_fits_exp_group_logit_ipw_svyglm_EO.csv", row.names = FALSE)

write.csv(top_fits_exp_group_logit_full_match_svyglm, file = "top_fits_exp_group_logit_full_match_svyglm_EO.csv", row.names = FALSE)







######################


# full match
########################


library(MatchIt)
library("marginaleffects")
library(sandwich)
fullmatch.out <- matchit(
  trt ~ age_months + sex,
  data = proof_of_concept_joined,
  method = "full", # optimal full match
  distance = 'glm',
  estimand = "ATE"
)




summary(fullmatch.out) # balanced
matched.data <- match.data(fullmatch.out)
mean(matched.data) # = 1


matched.data = matched.data %>% select(c('EOEC','subclass'))
matched.data$subclass
# sandbox
fit1 = lm(log(alpha_coh_Pz_C3 / (1-alpha_coh_Pz_C3)) ~ trt + EOEC + sex + age_months,data = matched.data, weights = weights)

avg_predictions(fit1, variables = "trt",
                vcov = ~subclass,
                wts = "weights",
                by = "trt")


design <- svydesign(id =~ 1, weights =~ weights, data = matched.data)
res <- svyglm(log(alpha_coh_Pz_C3 / (1-alpha_coh_Pz_C3)) ~ trt + EOEC + sex + age_months,
              design = design,family = gaussian)
summary(res)


hist(top_fits_exp_group_logit_full_match_svyglm$p.value)





















#################
### AIPW
#################


library(sandwich)
options(digits = 8)
aipw <- function(phenotype="bp_delta_Fp1", exposure="trt", covariates=c("sex", "age_months","EOEC"), data=proof_of_concept_joined, coh_mod_choice = 'logit') {
  # phenotype: a string that is a phenotype name
  # exposure: a string this is a exposure name 
  # covariates: an array of covariate variables
  
  
  data <- proof_of_concept_joined %>% drop_na({{phenotype}}) %>% 
    select({{phenotype}}, {{exposure}}, all_of(covariates))
  all_trt_0 = data %>% filter(trt == 0)
  all_trt_1 = data %>% filter(trt == 1)
  # create outcome model: phenotype ~ trt + sex + age_months + EOEC
  covariate_string <- paste(covariates, collapse="+")
  if (phenotype != "alpha_presence") {
    
    # check if outcome is spectral coherence
    if (grepl("coh", phenotype) | grepl("bp", phenotype)) {
      
      # chose to model using beta regression
      if (coh_mod_choice == 'beta'){
        print('beta')
        mod_string <- sprintf('%s ~ %s', phenotype,covariate_string)
        mod_0 <- betareg(as.formula(mod_string), data = all_trt_0, weights = all_trt_0$sw_weights)
        mod_1 <- betareg(as.formula(mod_string), data = all_trt_0, weights = all_trt_0$sw_weights)
        mu0 <- predict(mod_0,type = 'response', newdata = data )
        mu1 <- predict(mod_1, type = 'response', newdata = data)
      }
      
      # chose to model using logit transformation
      if (coh_mod_choice == 'logit') {
        print('logit')
        mod_string <- sprintf('log(%s/(1-%s)) ~ %s', phenotype, phenotype, covariate_string)
        mod_0 <- lm(as.formula(mod_string),data = all_trt_0, weights = all_trt_0$sw_weights)
        mod_1 <- lm(as.formula(mod_string),data = all_trt_1, weights = all_trt_1$sw_weights)
        
        mu0 <- predict(mod_0,type = 'response', newdata = data)
        
        mu1 <- predict(mod_1, type = 'response', newdata = data)
        
        
        # get the ate
        dr <- mean(data[[exposure]] * (log(data[[phenotype]]/(1-data[[phenotype]])) - mu1)/data$ps + mu1, na.rm = T) -
          mean((1 - data[[exposure]])*(log(data[[phenotype]]/(1-data[[phenotype]])) - mu0) / (1-data$ps) + mu0, na.rm = T)
        
        # generate p-value
        print('got here')
        vcov_robust <- vcovHC(mod_1, type = "HC3")
        
        se_robust <- sqrt(t(crossprod(crossprod(data.matrix(model.matrix(mod_1, newdata = data)), vcov_robust), t(data.matrix(model.matrix(mod_1, newdata = data))))))
        
        
       
        
      } 
      
      
    }
    
    
    else if (grepl("sample", phenotype) | grepl("sd_", phenotype) | grepl("kurt_", phenotype)){
      print('log outcome')
      mod_string <- sprintf('log(%s) ~ %s', phenotype, covariate_string)
      mod_0 <- lm(as.formula(mod_string),data = all_trt_0, weights = all_trt_0$sw_weights)
      mod_1 <- lm(as.formula(mod_string),data = all_trt_1, weights = all_trt_1$sw_weights)
      mu0 <- predict(mod_0,type = 'response', newdata = data)
      mu1 <- predict(mod_1, type = 'response', newdata = data)
      # get the ate
      dr <- mean(data[[exposure]] * (log(data[[phenotype]]) - mu1)/data$ps + mu1, na.rm = T) -
        mean((1 - data[[exposure]])*(log(data[[phenotype]]) - mu0) / (1-data$ps) + mu0, na.rm = T)
      
      # generate p-value
      se <- sqrt(var(data[[exposure]] * (log(data[[phenotype]]) - mu1)/data$ps + mu1, na.rm = T) /nrow(data) +
        var((1 - data[[exposure]])*(log(data[[phenotype]]) - mu0) / (1-data$ps) + mu0, na.rm = T) / nrow(data))
      
    }
    
    # if other features that is not spectral coherence, relative power, or sample entropy
    else {
      print('regular regression')
      mod_string <- sprintf('%s ~ %s', phenotype, covariate_string)
      mod_0 <- lm(as.formula(mod_string),data = all_trt_0, weights = all_trt_0$sw_weights)
      mod_1 <- lm(as.formula(mod_string),data = all_trt_1, weights = all_trt_1$sw_weights)
      mu0 <- predict(mod_0,type = 'response', newdata = data)
      mu1 <- predict(mod_1, type = 'response', newdata = data)
      
      # get the ate
      dr <- mean(data[[exposure]] * (data[[phenotype]] - mu1)/data$ps + mu1, na.rm = T) -
        mean((1 - data[[exposure]])*(data[[phenotype]] - mu0) / (1-data$ps) + mu0, na.rm = T)
    
      # generate p-value
      se <- sqrt(var(data[[exposure]] * (data[[phenotype]] - mu1)/data$ps + mu1, na.rm = T) /nrow(data) +
        var((1 - data[[exposure]])*(data[[phenotype]] - mu0) / (1-data$ps) + mu0, na.rm = T) / nrow(data))
    
     
      
      }
    
    
    
  }
  
  
  else{
    print('Logistic regression')
    mod_string <- sprintf('%s ~ %s', phenotype, covariate_string)
    # the above formats formula as a string 
    mod_0 <- glm(as.formula(mod_string), data = all_trt_0, family = "binomial",weights = all_trt_0$sw_weights)
    mod_1 <- glm(as.formula(mod_string), data = all_trt_1, family = "binomial",weights = all_trt_1$sw_weights)
    mu0 <- predict(mod_0,type = 'link', newdata = data)
    mu1 <- predict(mod_1, type = 'link', newdata = data)
    
    # get the ate
    dr <- mean(data[[exposure]] * (log(data[[phenotype]]/(1-data[[phenotype]])) - mu1)/data$ps + mu1,na.rm = T) -
      mean((1 - data[[exposure]])*(log(data[[phenotype]]/(1-data[[phenotype]])) - mu0) / (1-data$ps) + mu0, na.rm = T)
    
    # generate p-value
    se <- sqrt(var(data[[exposure]] * (log(data[[phenotype]]/(1-data[[phenotype]])) - mu1)/data$ps + mu1, na.rm = T) /nrow(data) +
      var((1 - data[[exposure]])*(log(data[[phenotype]]/(1-data[[phenotype]])) - mu0) / (1-data$ps) + mu0, na.rm = T) / nrow(data))
   
    
  }
  
 
  p_value <- 2 * (1 - pnorm(abs(dr/se)))
  df <- data.frame(estimate = dr, se = se, p.value = p_value)
  return(df)
}


list_of_fits <- list()



## Running through all phenotypes AIPW

# loop through all the outcomes in feature list
for (phenotype in unique(feature_list)) {
  summary_tibble <- aipw(phenotype, exposure="trt", data=proof_of_concept_joined, coh_mod_choice = 'logit')
  summary_tibble$outcome <- phenotype
  print(phenotype)
  sample_sizes <- proof_of_concept_joined %>% 
    filter(!is.na(.data[[phenotype]])) %>% 
    count(trt)
  
  summary_tibble$n_ASD <- sample_sizes$n[which(sample_sizes$trt == 1)]
  summary_tibble$n_control <- sample_sizes$n[which(sample_sizes$trt == 0)]
  
  
  list_of_fits <- c(list_of_fits,list(summary_tibble))
}


all_fits = bind_rows(list_of_fits)



bonferroni_pvalues <- p.adjust(all_fits$p.value, method = "bonferroni")

all_fits$bonf_p.value <- bonferroni_pvalues

fdr_adjusted_pvalues <- p.adjust(all_fits$p.value, method = "fdr")

all_fits$q.value <- fdr_adjusted_pvalues

top_fits_exp_group_logit_aipw <- all_fits %>% arrange(p.value) 

############## test with textbook data #################################
data = read.csv("./learning_mindset.csv")


# Define the categorical and continuous feature columns
categ <- c("ethnicity", "gender", "school_urbanicity")
cont <- c("school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size")

# Create a new dataset with dummy variables for categorical features
data_with_categ <- cbind(
  data[, !names(data) %in% categ], # dataset without categorical features
  as.data.frame(model.matrix(~., data = data[categ])) # categorical features converted to dummies
)

# Rename the column names to remove the intercept term
colnames(data_with_categ)[-(1:length(cont))] <- gsub(".*\\.", "", colnames(data_with_categ)[-(1:length(cont))])

# Print the shape of the new dataset
print(dim(data_with_categ))

data_with_categ$ethnicity = as.factor(data_with_categ$ethnicity)
data_with_categ$gender = as.factor(data_with_categ$gender)
data_with_categ$school_urbanicity = as.factor(data_with_categ$school_urbanicity)




results = t.test(c(1,2,3,4,5),c(1,8,9,8,7))

results$estimate


# Propensity model
propensity_model <- glm(intervention ~ success_expect + frst_in_family + school_mindset +
                          school_achievement + school_ethnic_minority + school_poverty +
                          school_size + ethnicity + gender + school_urbanicity, data = data_with_categ, family = "binomial")






data$ps <- predict(propensity_model,type='response') 

set.seed(42)

data$trt = data$intervention
data$weights <- ifelse(data$trt == 1, 1/data$ps, 1/(1-data$ps))

data$denominator <- ifelse(data$trt == 1, data$ps, 1-data$ps)

data$numerator <-  ifelse(data$trt == 1, mean(data$trt == 1), mean(data$trt == 0))

data$sw_weights <- data$numerator / data$denominator




aipw(phenotype = "achievement_score",exposure = 'trt',covariates = 
       c("success_expect", "frst_in_family", "school_mindset",
         "school_achievement", "school_ethnic_minority", "school_poverty",
         "school_size", "ethnicity", "gender", "school_urbanicity"), data =data)





summary(association(phenotype = "achievement_score",exposure = 'trt',covariates = 
              c("success_expect", "frst_in_family", "school_mindset",
                "school_achievement", "school_ethnic_minority", "school_poverty",
                "school_size", "ethnicity", "gender", "school_urbanicity"), data =data))

##################################################################################





#########
# Perform for EO and EC seperately
#########

eoec_0_data <- subset(proof_of_concept_joined, EOEC == 'EC')
eoec_1_data <- subset(proof_of_concept_joined, EOEC == 'EO')

# Perform matching on EOEC = 0 data
eoec_0_match.out <- matchit(
  trt ~ age_months + sex,
  data = eoec_0_data,
  method = "full",
  distance = "glm",
  estimand = "ATE"
)

summary(eoec_0_match.out)
eoec_0_matched.data <- match.data(eoec_0_match.out)

# Create empty list that holds all models
list_of_fits <- list()

# loop through all the outcomes in feature list
for (phenotype in unique(feature_list)) {
  assoc_model <- association(phenotype, exposure="trt", data=eoec_0_matched.data, coh_mod_choice = 'logit',
                             covariates=c("sex", "age_months"))
  summary_tibble <- tidy(assoc_model)
  summary_tibble$outcome <- phenotype
  print(phenotype)
  sample_sizes <- eoec_0_matched.data %>% 
    filter(!is.na(.data[[phenotype]])) %>% 
    count(trt)
  
  summary_tibble$n_ASD <- sample_sizes$n[which(sample_sizes$trt == 1)]
  summary_tibble$n_control <- sample_sizes$n[which(sample_sizes$trt == 0)]
  
  
  list_of_fits <- c(list_of_fits,list(summary_tibble))
}


all_fits = bind_rows(list_of_fits) %>% filter(term== 'trt')



bonferroni_pvalues <- p.adjust(all_fits$p.value, method = "bonferroni")

all_fits$bonf_p.value <- bonferroni_pvalues

fdr_adjusted_pvalues <- p.adjust(all_fits$p.value, method = "fdr")

all_fits$q.value <- fdr_adjusted_pvalues






# sort the data frame by the q-value column in ascending order
top_fits_EC_exp_group_logit_optimal_full_match <- all_fits %>% arrange(p.value) %>% head(50)






# Perform matching on EOEC = 1 data
eoec_1_match.out <- matchit(
  trt ~ age_months + sex,
  data = eoec_1_data,
  method = "full",
  distance = "glm",
  estimand = "ATE"
)

summary(eoec_1_match.out)
eoec_1_matched.data <- match.data(eoec_1_match.out)



# Create empty list that holds all models
list_of_fits <- list()

# loop through all the outcomes in feature list
for (phenotype in unique(feature_list)) {
  assoc_model <- association(phenotype, exposure="trt", data=eoec_1_matched.data, coh_mod_choice = 'logit',
                             covariates=c("sex", "age_months"))
  summary_tibble <- tidy(assoc_model)
  summary_tibble$outcome <- phenotype
  print(phenotype)
  sample_sizes <- eoec_1_matched.data %>% 
    filter(!is.na(.data[[phenotype]])) %>% 
    count(trt)
  
  summary_tibble$n_ASD <- sample_sizes$n[which(sample_sizes$trt == 1)]
  summary_tibble$n_control <- sample_sizes$n[which(sample_sizes$trt == 0)]
  
  
  list_of_fits <- c(list_of_fits,list(summary_tibble))
}


all_fits = bind_rows(list_of_fits) %>% filter(term== 'trt')



bonferroni_pvalues <- p.adjust(all_fits$p.value, method = "bonferroni")

all_fits$bonf_p.value <- bonferroni_pvalues

fdr_adjusted_pvalues <- p.adjust(all_fits$p.value, method = "fdr")

all_fits$q.value <- fdr_adjusted_pvalues






# sort the data frame by the q-value column in ascending order
top_fits_EO_exp_group_logit_optimal_full_match <- all_fits %>% arrange(p.value) %>% head(50)












































#################################
## Nearest neighbor propensity-score matching + Mann-Whitney U test
##
mannwhitney <- function(phenotype="bp_delta_Fp1", exposure="trt", covariates=c("sex", "age_months", "EOEC"), data=proof_of_concept_joined) {
  

  
  asd = matched_data %>% filter(!!sym(exposure) == 1) %>% 
    select(c(!!sym(phenotype))) %>% pull()
  control = matched_data %>% filter(!!sym(exposure) == 0) %>% 
             select(c(!!sym(phenotype))) %>% pull()
  

  # Perform Mann-Whitney U-test on matched data
  u_test <- wilcox.test(asd,control)
  
  # Compute sample size of treatment and control groups after matching
  n_treated <- sum(matched_data[[exposure]] == 1)
  n_control <- sum(matched_data[[exposure]] == 0)
  
  diff_in_median <- median(asd,na.rm = TRUE) - median(control, na.rm=TRUE)

  # Return outcome model and U-test results
  return(data.frame(
    phenotype = phenotype,
    p_value = u_test$p.value,
    n_treated = n_treated,
    n_control = n_control,
    diff_in_median = diff_in_median
  ))
}




results_df <- data.frame()

# Loop through all phenotypes in feature_list and run the function
for (phenotype in unique(feature_list)) {
  # Call the function and add results to the data frame
  results_df <- rbind(results_df, mannwhitney(phenotype=phenotype))
}

bonferroni_pvalues <- p.adjust(results_df$p_value, method = "bonferroni")

results_df$bonf_p.value <- bonferroni_pvalues

fdr_adjusted_pvalues <- p.adjust(results_df$p_value, method = "BH")

results_df$q.value <- fdr_adjusted_pvalues



top_fits_4 = results_df %>% arrange(q.value) %>% head(50)



library(ggplot2)

plot_matched <- function(phenotype="bp_delta_Fp1", exposure="trt", covariates=c("sex", "age_months", "EOEC"), data=proof_of_concept_joined) {
  
  # Drop all rows where there is NA in the phenotype column
  data <- data %>% drop_na({{phenotype}})
  
  # Create a propensity model: exposure ~ covariates
  ps_formula <- formula(paste0(exposure, " ~ ", paste(covariates, collapse="+")))
  ps_model <- glm(ps_formula, data=data, family="binomial")
  
  # Compute propensity scores and weights
  data$ps <- predict(ps_model, type="response")
  
  # Perform propensity score matching
  matchit_object <- matchit(ps_formula, data=data, method="nearest")
  matched_data <- match.data(matchit_object)
  
  # Extract the phenotype values for asd and control
  asd <- matched_data %>% filter(!!sym(exposure) == 1) %>% 
    select(c(!!sym(phenotype))) %>% pull()
  control <- matched_data %>% filter(!!sym(exposure) == 0) %>% 
    select(c(!!sym(phenotype))) %>% pull()
  
  # Create a data frame with the phenotype values and exposure status
  plot_data <- data.frame(
    phenotype = c(asd, control),
    exposure = rep(c("asd", "control"), c(length(asd), length(control)))
  )
  
  # Create the plot using ggplot2
  ggplot(plot_data, aes(x=exposure, y=phenotype)) +
    geom_jitter(width=0.2) +
    geom_violin(scale="width", width=0.8, fill="gray", alpha=0.5) +
    geom_boxplot(width=0.2, fill="white", alpha=0.8) +
    labs(x="Exposure", y=phenotype) +
    ggtitle(paste0("Boxplot and violin plot for ", phenotype)) +
    theme_bw()
}

pdf("matched_mann_whitney_top_50_cars.pdf")

# Loop over the phenotypes and create a plot for each one
for (i in top_fits_4$phenotype) {
  my_plot <- plot_matched(phenotype = i)
  print(my_plot)
}


# Close the PDF file
dev.off()
dev.cur()

# Switch to the graphics device that corresponds to the PDF file
dev.set(2)  # Replace 2 with the appropriate device number

# Try closing the PDF file again
dev.off()

