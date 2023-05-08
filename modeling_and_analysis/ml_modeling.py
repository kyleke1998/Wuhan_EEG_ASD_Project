import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score, roc_auc_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.stats import ttest_1samp
import os 
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from RENT import RENT
from sklearn.linear_model import SGDClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from mrmr import mrmr_classif
import mlflow
import lightgbm
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer




os.chdir("c:/Users/kylek/Dropbox (Partners HealthCare)/eeg_asd_project/Code")

FEATURES_TO_EXCLUDE = ["subject_x","subject_y",
                "file_name",
                      "trt","index_x","index_y","exp_group",
                      "weights","ps","exposure",
                      "CARS_Categorical","index","subject",
                      'numerator','denominator','sw_weights','subclass','distance','trt','ps','CARS_Categorical','exposure',
                      'weights','subclass','distance','numerator','denominator','sw_weights', 'Unnamed: 0','file_name','sex','age_months','subject','EOEC']



# Loading in data
df = pd.read_csv('../features/features_ml_eo.csv') # load data
#df['sex'] = df['sex'].apply(lambda x: 1 if x == 'm' else 0)




# get all the feature names
full_features = [col for col in df.columns if col not in FEATURES_TO_EXCLUDE]
full_features = full_features + ['sex','age_months']

y = df['trt']






def fit_elastic_net(train_index, test_index, X, y, best_C, best_l1_ratio):
    X_train, y_train = X[train_index], y[train_index]
    elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', C=best_C, l1_ratio=best_l1_ratio,
                                     class_weight='balanced', max_iter=1000)
    elastic_net.fit(X_train, y_train)
    return elastic_net.coef_


def repeated_elastic_net_select_features(X, y):
    """
    A parallel implementation of repeated elastic net for feature selection
    """
    # Perform 10-fold cross-validation to find the best hyperparameters for Elastic Net
    inner_cv = RepeatedStratifiedKFold(n_splits=10, random_state=3, n_repeats=1)

    param_grid = {'C': [0.01, 0.1, 1], 'l1_ratio': [0.75, 0.85, 1]}
    elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=1000)
    cv = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=inner_cv, n_jobs=-1, scoring='roc_auc')
    cv.fit(X, y)
    best_C = cv.best_params_['C']
    best_l1_ratio = cv.best_params_['l1_ratio']
    print(cv.best_score_)

    # Build 100 elastic net models with the best hyperparameters and store the coefficients
    n_models = 100
    
    sss = StratifiedShuffleSplit(n_splits=n_models, test_size=0.1, train_size=0.9,
                                                             random_state=0)
    coefs = Parallel(n_jobs=-1)(
        delayed(fit_elastic_net)(train_index, test_index, X, y, best_C, best_l1_ratio)
        for train_index, test_index in sss.split(X, y))
    coefs = np.squeeze(np.array(coefs))

    # Select features satisfying the three conditions
    # 1. Selected in 100% of the models
    n_models_with_feature = np.sum(coefs != 0, axis=0)
    # 2. Sign of the coefficient is the same in 100% of the models
    sign_agreement = np.mean(np.sign(coefs) == np.sign(coefs.mean(axis=0)), axis=0)
    # 3.mean(each coeff) not 0 -> p-value < 0.001
    t, p = ttest_1samp(coefs, 0, axis=0)
    selected_features = (n_models_with_feature >= n_models * 0.95) & (sign_agreement >= 0.95) & (p < 0.001)
    selected_features_indices = np.where(selected_features)[0]
    selected_features = df[full_features].columns[np.where(selected_features)[0]].to_list()
    print(selected_features)
    return selected_features







#############################################
#Greedy forward selection Logistic Regression#
#############################################
mlflow.set_experiment("Logistic Regression Greedy Forward Selection")


feature_list = []  

X = df.copy(deep=True)[full_features]
le = LabelEncoder()
X['sex'] = le.fit_transform(X['sex'])

coefficent_matrix = np.zeros((10,len(full_features)))

# Initialize the array to store the nested scores
nested_sensitivity = np.zeros(10)
nested_specificity = np.zeros(10)
nested_roc_auc = np.zeros(10)
nested_f1_macro = np.zeros(10)
nested_accuracy = np.zeros(10)
nested_C = np.zeros(10)


# Define the indices for the outer loop splits
outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
outer_cv_splits = outer_cv.split(X, y)


# Loop over the outer loop splits
for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):

    # Split the data into training/validation and testing sets
    X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
    X_test, y_test = X.loc[test_idx,:], y[test_idx]

    features = X_train_val.columns 

    steps = [('mean_impute', SimpleImputer(strategy='mean'))]
    #steps = [('impute', KNNImputer(n_neighbors=3))]
    pipeline = Pipeline(steps=steps)

    # Apply the pipeline on the training/validation set
    X_train_val_transformed = pipeline.fit_transform(X_train_val)


    # Apply the pipeline on the testing set
    X_test_transformed = pipeline.transform(X_test)



    # Define the indices for the inner loop splits
    inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

    # Define the estimator and the parameter grid for GridSearchCV
    lr = LogisticRegression(class_weight='balanced',max_iter=1000)
    param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100]}

    clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    # Initialize the SequentialFeatureSelector with cv=None
    sfs = SequentialFeatureSelector(clf, n_features_to_select=10, cv=splitter,n_jobs=-1)


    # Fit the SequentialFeatureSelector on the train data
    sfs.fit(X_train_val_transformed , y_train_val)

    # Get selected features
    selected_features = sfs.get_feature_names_out(input_features=features)
    #selected_features = benchmark_features



    
    feature_list.append(selected_features)

    selected_features = np.append(selected_features,['sex','age_months'])
    selected_features = list(set(selected_features))

    # subset to the selected features
    
    # get index of selected features
    selected_feature_indices = [X_train_val.columns.get_loc(feature) for feature in selected_features]
    
    X_train_val_transformed = X_train_val_transformed[:,selected_feature_indices]
    X_test_transformed = X_test_transformed[:,selected_feature_indices]
    
    
    """
    X_train_val = X_train_val[selected_features]
    X_test = X_test[selected_features]

    # Apply the pipeline on the training/validation set
    X_train_val_transformed = pipeline.fit_transform(X_train_val)


    # Apply the pipeline on the testing set
    X_test_transformed = pipeline.transform(X_test)
    """
  

    # reinstaniate a gridsearchcv
    clf.fit(X_train_val_transformed, y_train_val)
  
    # update coefficent matrix
    coefs = clf.best_estimator_.coef_[0]

    # update the coefficient matrix
    coefficent_matrix[j, selected_feature_indices] = coefs  
   
 
    y_pred = clf.predict(X_test_transformed)
    y_pred_proba = clf.predict_proba(X_test_transformed)[:,1]


    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the scores in the array
    nested_sensitivity[j] = sensitivity
    nested_specificity[j] = specificity
    nested_roc_auc[j] = roc_auc
    nested_f1_macro[j] = f1_macro
    nested_accuracy[j] = accuracy


    print("Finished ",j)

with mlflow.start_run(run_name='greedy_forward_lr',nested=True):   

        # Log the features used
        mlflow.log_param("C", clf.best_params_['C'])

        # Compute the mean and the standard deviation of the nested scores
        mean_nested_sensitivity = nested_sensitivity.mean()
        mlflow.log_metric('sensitivity', mean_nested_sensitivity)
        mean_nested_specificity = nested_specificity.mean()
        mlflow.log_metric('specificity', mean_nested_specificity)
        mean_nested_roc_auc = nested_roc_auc.mean()
        mlflow.log_metric('roc_auc', mean_nested_roc_auc)
        mean_nested_f1_macro = nested_f1_macro.mean()
        mlflow.log_metric('f1_macro', mean_nested_f1_macro)
        mean_nested_accuracy = nested_accuracy.mean()
        mlflow.log_metric('accuracy', mean_nested_accuracy)

mlflow.end_run()
# save coefficent matrix
pd.DataFrame(coefficent_matrix).to_csv('coefficeint_matrix_forward_greedy_lr.csv',index=False)
# save feature list 
pd.DataFrame(feature_list).to_csv('selected_features_forward_greedy_lr.csv',index=False)


################################################
#Greedy forward selection Logistic Regression Ends#                          
#################################################





##################################################
# Repeated Elastic Net Feature Selection Nested CV#
##################################################

selected_feature_list = []  



numeric_cols = [col for col in full_features if col not in ['sex','alpha_presence']]

# Initialize the array to store the nested scores
nested_sensitivity = np.zeros(10)
nested_specificity = np.zeros(10)
nested_roc_auc = np.zeros(10)
nested_f1_macro = np.zeros(10)
nested_accuracy = np.zeros(10)
nested_C = np.zeros(10)

nested_sensitivity_2 = np.zeros(10)
nested_specificity_2 = np.zeros(10)
nested_roc_auc_2 = np.zeros(10)
nested_f1_macro_2 = np.zeros(10)
nested_accuracy_2 = np.zeros(10)
nested_max_depth = np.zeros(10)
nested_num_leaves = np.zeros(10)
nested_n_estimator = np.zeros(10)

# Define the indices for the outer loop splits
outer_cv = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
outer_cv_splits = outer_cv.split(features_df, y)




# Loop over the outer loop splits
for j, (train_val_idx, test_idx) in enumerate(outer_cv.split(features_df, y)):

    # Split the data into training/validation and testing sets
    X_train_val, y_train_val = features_df.loc[train_val_idx,:], y[train_val_idx]
    X_test, y_test = features_df.loc[test_idx,:], y[test_idx]


    numeric_cols = [col for col in full_features if col not in ['sex', 'alpha_presence']]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('passthrough', 'passthrough', ['sex', 'alpha_presence'])  # Add a passthrough transformer
        ])

    steps = [('preprocessor',preprocessor), ('impute', KNNImputer(n_neighbors=3))]
    pipeline = Pipeline(steps=steps)

    # Apply the pipeline on the training/validation set
    X_train_val = pipeline.fit_transform(X_train_val)
    

    # Apply the pipeline on the testing set
    X_test = pipeline.transform(X_test)

    # Define the indices for the inner loop splits
    selected_features = repeated_elastic_net_select_features(X_train_val, y_train_val.to_numpy())

    # append selected features to the list of selected features since there will be 10 sets of selected features
    selected_feature_list.append(selected_features)

    # force sex and age into the selection
    selected_features = list(set(selected_features + ['sex','age_months']))

    # get selected feature indicies
    selected_feature_indices = [features_df.columns.get_loc(feature) for feature in selected_features]




    X_train_val = X_train_val[:,selected_feature_indices]
    X_test = X_test[:,selected_feature_indices]

    #################################
    # *Tune Logistic Regression on Selected Features
    inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

        
    # Define the estimator and the parameter grid for GridSearchCV
    lr = LogisticRegression(class_weight='balanced',max_iter=1000)
    param_grid = {'C': [0.0001,0.001, 0.01, 0.1,1]}

    clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
    
    # Fit the estimator on the training/validation set and calibrate the predictions
    clf.fit(X_train_val, y_train_val)

 
 
    y_pred = clf.predict(X_test)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(roc_auc)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    # Store the scores in the array
    nested_sensitivity[j] = sensitivity
    nested_specificity[j] = specificity
    nested_roc_auc[j] = roc_auc
    nested_f1_macro[j] = f1_macro
    nested_accuracy[j] = accuracy
    nested_C[j] = clf.best_params_['C']


    #################################
    # *Tune Lightgbm on Selected Features*
    param_grid = {
            'max_depth': [3,5,-1],
            'num_leaves': [10],
            'n_estimators': [30,40,50]
            }

    lgb = lightgbm.LGBMClassifier(n_jobs=-1,random_state=42,is_unbalance=True)

    clf = GridSearchCV(estimator=lgb, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')

    # Fit the estimator on the training/validation set and calibrate the predictions
    clf.fit(X_train_val, y_train_val)



    
    y_pred = clf.predict(X_test)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(roc_auc)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)


    # Store the scores in the array
    nested_sensitivity_2[j] = sensitivity
    nested_specificity_2[j] = specificity
    nested_roc_auc_2[j] = roc_auc
    nested_f1_macro_2[j] = f1_macro
    nested_accuracy_2[j] = accuracy
    nested_num_leaves[j] = clf.best_params_['num_leaves']
    nested_max_depth[j] = clf.best_params_['max_depth']
    nested_n_estimator[j] = clf.best_params_['n_estimators']

    print("Finished ",j)


mlflow.set_experiment("Nested CV Repeated Elastic Net Feature Selection")





with mlflow.start_run(run_name='Logistic Regression L2',nested=True):   

        
        mlflow.log_param('features', selected_feature_list)

        # Compute the mean and the standard deviation of the nested scores
        mean_nested_sensitivity = nested_sensitivity.mean()
        mlflow.log_metric('mean sensitivity', mean_nested_sensitivity)
        mlflow.log_metric('SD sensitivity', nested_sensitivity.std())
        mean_nested_specificity = nested_specificity.mean()
        mlflow.log_metric('mean specificity', mean_nested_specificity)
        mlflow.log_metric('SD specificity', nested_specificity.std())
        mean_nested_roc_auc = nested_roc_auc.mean()
        mlflow.log_metric('mean roc_auc', mean_nested_roc_auc)
        mlflow.log_metric('SD roc_auc', nested_roc_auc.std())
        mean_nested_f1_macro = nested_f1_macro.mean()
        mlflow.log_metric('mean f1_macro', mean_nested_f1_macro)
        mlflow.log_metric('SD f1_macro', nested_f1_macro.std())
        mean_nested_accuracy = nested_accuracy.mean()
        mlflow.log_metric('mean accuracy', mean_nested_accuracy)
        mlflow.log_metric('SD accuracy', nested_accuracy.std())

mlflow.end_run()
    

###########################################################################################





benchmark_features = ['gamma_coh_P4_F4','kurt_Pz', 'kurt_P3','gamma_coh_P4_F3', 'beta_coh_P4_F4',       
                       'kurt_C4', 'gamma_coh_P4_C3','beta_coh_P4_F3','skew_F7',
                       'alpha_coh_P4_F4','beta_coh_P4_C3','sex','age_months']








########################
# MRMR Feature Selection#
#######################
feature_list = [] 

X = df.copy(deep=True)[full_features]
le = LabelEncoder()
X['sex'] = le.fit_transform(X['sex'])
numeric_cols = [col for col in full_features if col not in ['sex_male','alpha_presence']]

# Initialize the array to store the nested scores
nested_sensitivity = np.zeros(10)
nested_specificity = np.zeros(10)
nested_roc_auc = np.zeros(10)
nested_f1_macro = np.zeros(10)
nested_accuracy = np.zeros(10)
nested_C = np.zeros(10)

# Define the indices for the outer loop splits


outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
outer_cv_splits = outer_cv.split(X, y)



# Loop over the outer loop splits
for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):

    # Split the data into training/validation and testing sets
    X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
    X_test, y_test = X.loc[test_idx,:], y[test_idx]

    
    # Perform maximum relevance minimum redundancy feature selection
    #selected_features = mrmr_classif(X_train_val, y_train_val, K = 10)
    #selected_features = list(set(selected_features + ['sex','age_months']))
    selected_features = benchmark_features
    # Get selected feature indices,
    #selected_feature_indices = [X_train_val.columns.get_loc(feature) for feature in selected_features]
    X_train_val = X_train_val[selected_features]
    X_test = X_test[selected_features]
   



    feature_list.append(selected_features)
    
    numeric_cols = [col for col in X_train_val.columns if col not in ['sex', 'alpha_presence']]
    

    steps = [('impute', KNNImputer(n_neighbors=3))]
    pipeline = Pipeline(steps=steps)

    # Apply the pipeline on the training/validation set
    X_train_val = pipeline.fit_transform(X_train_val)


    # Apply the pipeline on the testing set
    X_test = pipeline.transform(X_test)

    print(X_test.shape)
   

  
    # Define the indices for the inner loop splits
    
    inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

    # Define the estimator and the parameter grid for GridSearchCV
    lr = LogisticRegression(class_weight='balanced',max_iter=1000)
    param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100]}

    clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
    clf.fit(X_train_val, y_train_val)
    
    nested_C = clf.best_params_['C']

    y_pred = clf.predict(X_test)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    # Store the scores in the array
    nested_sensitivity[j] = sensitivity
    nested_specificity[j] = specificity
    nested_roc_auc[j] = roc_auc
    nested_f1_macro[j] = f1_macro
    nested_accuracy[j] = accuracy

    print("Finished ",j)



from collections import Counter

# Flatten the nested list into a single list
flat_list = [feature for sublist in feature_list for feature in sublist]

# Count the occurrences of each feature in the list
feature_counts = Counter(flat_list)











#################################
# Single Greedy Forward Selection
#################################


inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=1)

# Define the estimator and the parameter grid for GridSearchCV
lr = LogisticRegression(class_weight='balanced',max_iter=1000)
param_grid = {'C': [0.0001,0.001, 0.01, 0.1]}

clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# Initialize the SequentialFeatureSelector with cv=None
sfs = SequentialFeatureSelector(clf, n_features_to_select=10, cv=splitter,n_jobs=-1)


# Fit the SequentialFeatureSelector on the train data
sfs.fit(X_train_val , y_train_val)

selected_features = sfs.get_feature_names_out(input_features=full_features)

selected_features = np.append(selected_features,['sex','age_months'])

# get index of selected features
selected_feature_indices = [df[full_features].columns.get_loc(feature) for feature in selected_features]

X_train_val = X_train_val[:,selected_feature_indices]
X_test = X_test[:,selected_feature_indices]
sfs.estimator.fit(X_train_val, y_train_val)
y_pred = sfs.estimator.best_estimator_.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred)





