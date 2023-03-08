library(twang)
library(tidyverse)
library(broom)

metadata = read.csv("C:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/eeg_data/wuhan_study_clinical_data.csv")
metadata = metadata %>% filter(((EOEC=='EO') | (EOEC=='EC')) & ((exp_group=='ASD') | (exp_group=='No ASD')))
metadata$EOEC = as.factor(metadata$EOEC)
metadata$sex = as.factor(metadata$sex)
metadata$exp_group = as.factor(metadata$exp_group)
metadata$subject_id = as.integer(metadata$subject_id)
metadata$trt = ifelse(metadata$exp_group=='ASD',1,0)

metadata_iq_filtered = metadata %>% filter(!is.na(metadata$IQ)) 





metadata











# Loading in feature set of spectral power features as proof of concept     ###### 
power_features = read.csv("C:/Users/kylek/Dropbox (Partners HealthCare)/kevin_root/spectral_power_2x14/wd_data/spec_pwr_2x14.csv")
proof_of_concept_joined = metadata %>% select(c(sex,age_months,exp_group,subject_id,trt)) %>%
  right_join(y=power_features,by='subject_id')
  


# gradient boosted trees propensity model
ps.gbm = ps(trt ~ age_months + EOEC + sex + IQ,
                    data = metadata_iq_filtered,
            estimand = "ATE",
            verbose = FALSE)


hist(metadata$age_months)

# convergence diagnostics
plot(ps.gbm)


# balanced diagnostics
bal.table(ps.gbm)

#
plot(ps.gbm, plots=3)



summary(ps.gbm$gbm.obj,plot=T)


summary(metadata$exp_group)



# Logistic Regression propensity mode
ps.logit <- glm(trt ~ age_months + sex + IQ,
    data = metadata_iq_filtered,family='binomial')
summary(ps.logit)


metadata_iq_filtered = metadata %>% filter(!is.na(metadata$IQ)) 
metadata_iq_filtered $ps <- predict(ps.logit,type='response') 

bal.logit <- dx.wts(x = metadata_iq_filtered$ps,
                     data=metadata_iq_filtered,
                     vars=c("age_months","sex"),
                     treat.var="trt",
                    x.as.weights=F,
                     perm.test.iters=0, estimand = "ATE")

bal.table(bal.logit)



hist(log(power_features$Delta_TAL_BR))


################################################################################################################
# Run many regressions
################################################################################################################


# Propensity scores
ps.logit <- glm(trt ~ age_months + sex,
                data = proof_of_concept_joined,family='binomial')



proof_of_concept_joined$ps <- predict(ps.logit,type='response') 
proof_of_concept_joined$weights <- ifelse(proof_of_concept_joined$trt == 1, 1/proof_of_concept_joined$ps,
                                          1/(1-proof_of_concept_joined$ps))






# function to run a single outcome model

association <- function(phenotype="Delta_TAL_BR", exposure="trt", covariates=c("sex", "age_months"), data) {
  # phenotype: a string that is a phenotype name
  # exposure: a string this is a exposure name 
  # covariates: an array of covariate variables
  # dsn: the survey design object
  covariate_string <- paste(covariates, collapse="+")
  mod_string <- sprintf('log10(%s) ~ %s + %s', phenotype, exposure, covariate_string)
  # the above formats formula as a string 
  mod <- lm(as.formula(mod_string),data = data,weights = proof_of_concept_joined$weights)
  return(mod)
}

# feature list
feature_list = colnames(power_features[, -which(names(power_features) %in% c("subject_id"))])


# Create empty list that holds all models
list_of_fits <- list()


# loop through all the outcomes in feature list
for (phenotype in unique(feature_list)) {
  assoc_model <- association(phenotype, exposure="trt", data=proof_of_concept_joined)
  summary_tibble <- tidy(assoc_model)
  summary_tibble$outcome <- phenotype
  list_of_fits <- c(list_of_fits,list(summary_tibble))
}


all_fits = bind_rows(list_of_fits) %>% filter(term== 'trt')
hist(all_fits$p.value)
