library(CausalGAM)



casualgam_ate <- function(phenotype="bp_delta_Fp1", exposure="trt", covariates=c("sex", "age_months","EOEC"), data=proof_of_concept_joined, coh_mod_choice = 'logit') {
  # phenotype: a string that is a phenotype name
  # exposure: a string this is a exposure name 
  # covariates: an array of covariate variables
  
  data <- data  %>% drop_na({{phenotype}}) %>% 
    select({{phenotype}}, {{exposure}}, all_of(covariates))
  # create outcome model: phenotype ~ trt + sex + age_months + EOEC
  covariate_string <- paste(covariates, collapse="+")
  
  # create pscore model: trt ~ sex + age_months + EOEC
  pscore_formula <- paste(exposure, covariate_string, sep="~")
  
  if (phenotype != "alpha_presence") {
    
    # check if outcome is spectral coherence
    if (grepl("coh", phenotype) | grepl("bp", phenotype)) {
      
      # chose to model using logit transformation
      if (coh_mod_choice == 'logit') {
        print('logit')
        mod_string <- sprintf('log(%s/(1-%s)) ~ %s + %s', phenotype, phenotype, exposure, covariate_string)
        mod <- estimate.ATE(pscore.formula = as.formula(pscore_formula),
                            pscore.family = binomial,
                            outcome.formula.t = as.formula(mod_string),
                            outcome.formula.c = as.formula(mod_string),
                            outcome.family = gaussian,
                            treatment.var = sprintf('%s',exposure),
                            data = data,
                            divby0.action = "t",
                            divby0.tol = 0.001,
                            var.gam.plot = FALSE,
                            nboot = 0)
      } 
      
      
    }
    
    
    else if (grepl("sample", phenotype) | grepl("sd_", phenotype) | grepl("kurt_", phenotype)){
      print('log outcome')
      mod_string <- sprintf('log(%s) ~ %s + %s', phenotype, exposure,covariate_string)
      mod <- estimate.ATE(pscore.formula = as.formula(pscore_formula),
                          pscore.family = binomial,
                          outcome.formula.t = as.formula(mod_string),
                          outcome.formula.c = as.formula(mod_string),
                          outcome.family = gaussian,
                          treatment.var = sprintf('%s',exposure),
                          data = data,
                          divby0.action = "t",
                          divby0.tol = 0.001,
                          var.gam.plot = FALSE,
                          nboot = 0)
    } 
      
    
    
    # if other features that is not spectral coherence, relative power, or sample entropy
    else {
      print('regular regression')
      mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
      mod <- estimate.ATE(pscore.formula = as.formula(pscore_formula),
                          pscore.family = binomial,
                          outcome.formula.t = as.formula(mod_string),
                          outcome.formula.c = as.formula(mod_string),
                          outcome.family = gaussian,
                          treatment.var = exposure,
                          data = data,
                          divby0.action = "t",
                          divby0.tol = 0.001,
                          var.gam.plot = FALSE,
                          nboot = 0)
    }
    
    # the above formats formula as a string 
    
}
  else {
    print('logistic regression')
    mod_string <- sprintf('%s ~ %s + %s', phenotype, exposure, covariate_string)
    mod <- estimate.ATE(pscore.formula = as.formula(pscore_formula),
                        pscore.family = binomial,
                        outcome.formula.t = as.formula(mod_string),
                        outcome.formula.c = as.formula(mod_string),
                        outcome.family = binomial,
                        treatment.var = exposure,
                        data = data,
                        divby0.action = "t",
                        divby0.tol = 0.001,
                        var.gam.plot = FALSE,
                        nboot = 0)
  }

  ATE.AIPW.hat <- mod$ATE.AIPW.hat
  ATE.AIPW.sand.SE <- mod$ATE.AIPW.sand.SE
  ATE.IPW.SE <- mod$ATE.IPW.asymp.SE
  ATE.IPW.hat <- mod$ATE.IPW.hat


  p_value_aipw <- 2 * pnorm(-abs(ATE.AIPW.hat / ATE.AIPW.sand.SE))
  p_value_ipw <- 2 * pnorm(-abs(ATE.IPW.hat / ATE.IPW.SE))
  
  results <- data.frame(ATE.AIPW.hat = ATE.AIPW.hat,
                        ATE.AIPW.sand.SE = ATE.AIPW.sand.SE,
                        p_value_aipw  = p_value_aipw ,
                        ATE.IPW.SE  = ATE.IPW.SE,
                        ATE.IPW.hat = ATE.IPW.hat,
                        p_value_ipw = p_value_ipw)
  
  return(results)
  
}




# feature list
feature_list = colnames(proof_of_concept_joined[, -which(names(proof_of_concept_joined) %in% 
                                                           c("subject_x","subject_y","sex","age_months","EOEC","file_name","trt","index_x","index_y","exp_group","weights","ps","exposure","CARS_Categorical","index","subject",
                                                             'numerator','denominator','sw_weights','subclass','distance'))])


# Create empty list that holds all models
list_of_fits <- list()


for (phenotype in feature_list[147:length(feature_list)]) {
  print(phenotype)
  
  exposure = "trt"
  phenotype=phenotype
  covariates=c("sex", "age_months","EOEC")
  data <- proof_of_concept_joined %>% drop_na({{phenotype}}) %>% 
    select({{phenotype}}, {{exposure}}, all_of(covariates))
  
  summary_tibble <- casualgam_ate(phenotype, exposure="trt", data=data, coh_mod_choice = 'logit')
  summary_tibble$outcome <- phenotype
  sample_sizes <- proof_of_concept_joined %>% 
    filter(!is.na(.data[[phenotype]])) %>% 
    count(trt)
  
  summary_tibble$n_ASD <- sample_sizes$n[which(sample_sizes$trt == 1)]
  summary_tibble$n_control <- sample_sizes$n[which(sample_sizes$trt == 0)]
  
  
  list_of_fits <- c(list_of_fits,list(summary_tibble))
}


all_fits = bind_rows(list_of_fits) 


fdr_p_value_aipw <- p.adjust(all_fits$p_value_aipw, method = "fdr")

all_fits$fdr_p_value_aipw <- fdr_p_value_aipw


fdr_p_value_ipw <- p.adjust(all_fits$p_value_ipw, method = "fdr")

all_fits$fdr_p_value_ipw <- fdr_p_value_ipw

hist(all_fits$p_value_aipw)

top_fits_EC_exp_group_logit_aipw <- all_fits %>% arrange(all_fits$fdr_p_value_ipw) %>% head(50)

write.csv(top_fits_EC_exp_group_logit_aipw,"top_fits_exp_group_logit_aipw.csv")


exposure = "trt"
phenotype="bp_delta_Fp1"
covariates=c("sex", "age_months","EOEC")
data <- proof_of_concept_joined %>% drop_na({{phenotype}}) %>% 
  select({{phenotype}}, {{exposure}}, all_of(covariates))


# create outcome model: phenotype ~ trt + sex + age_months + EOEC
covariate_string <- paste(covariates, collapse="+")

# create pscore model: trt ~ sex + age_months + EOEC
pscore_formula <- paste(exposure, covariate_string, sep="~")

mod_string <- sprintf('log(%s/(1-%s)) ~ %s + %s', phenotype, phenotype, exposure, covariate_string)
print('got here')
mod <- estimate.ATE(pscore.formula = as.formula(pscore_formula),
                    pscore.family = binomial,
                    outcome.formula.t = as.formula(mod_string),
                    outcome.formula.c = as.formula(mod_string),
                    outcome.family = gaussian,
                    treatment.var = exposure,
                    data = data,
                    divby0.action = "t",
                    divby0.tol = 0.001,
                    var.gam.plot = FALSE,
                    nboot = 0)
casualgam_ate(phenotype = 'delta_coh_C4_F7', exposure="trt", covariates=c("sex", "age_months","EOEC"), data=proof_of_concept_joined, coh_mod_choice = 'logit')
phenotype = 'delta_coh_C4_F7'
data <- proof_of_concept_joined %>% drop_na({{phenotype}}) %>% 
  select({{phenotype}}, {{exposure}}, all_of(covariates))

hist(log(data$delta_coh_C4_F7/(1-data$delta_coh_C4_F7)))
