import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score, roc_auc_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
import lightgbm
from tqdm import tqdm
from sklearn.svm import SVC




FEATURES_TO_EXCLUDE = ["subject_x","subject_y",
                "file_name",
                      "trt","index_x","index_y","exp_group",
                      "weights","ps","exposure",
                      "CARS_Categorical","index","subject",
                      'numerator','denominator','sw_weights','subclass','distance','trt','ps','CARS_Categorical','exposure',
                      'weights','subclass','distance','numerator','denominator','sw_weights', 'Unnamed: 0','file_name','sex','age_months','subject','EOEC']



# Loading in data
df = pd.read_csv('features_ml_eo.csv') # load data

full_features = [col for col in df.columns if col not in FEATURES_TO_EXCLUDE]
# list of top 50 features 
#significance_ranking = pd.read_csv('top_fits_exp_group_logit_full_match_svyglm_EO.csv')
significance_ranking = pd.read_csv('top_fits_exp_group_logit_ipw_svyglm_eo.csv')
top_50_features = significance_ranking.outcome.tolist()
top_50_features_and_covariates = top_50_features + ['age_months','EOEC','sex']

# get features and the outcome
#features = df[top_50_features]
y = df['trt']




mlflow.set_experiment("Logistic Regression Nested CV IPW EO")



##################################
# Logistic Regression 
###################################
def logistic_regression_nested_cv(feature_list,X,y,num_run):

    X = X.copy(deep=True)[feature_list]
    
    numeric_cols = [col for col in feature_list if col not in [ 'sex']]
    categorical_cols = [ 'sex']

    # Initialize the array to store the nested scores
    nested_sensitivity = np.zeros(10)
    nested_specificity = np.zeros(10)
    nested_roc_auc = np.zeros(10)
    nested_f1_macro = np.zeros(10)
    nested_accuracy = np.zeros(10)

    # Define the indices for the outer loop splits
    outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    outer_cv_splits = outer_cv.split(X, y)

    # Loop over the outer loop splits
    for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):
       
        # Split the data into training/validation and testing sets
        X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
        X_test, y_test = X.loc[test_idx,:], y[test_idx]
        

        preprocessor = ColumnTransformer(
        transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
        ])

        steps = [('preprocessor',preprocessor), ('impute', KNNImputer(n_neighbors=3))]
        pipeline = Pipeline(steps=steps)

        # Apply the pipeline on the training/validation set
        X_train_val = pipeline.fit_transform(X_train_val)

        # Apply the pipeline on the testing set
        X_test = pipeline.fit_transform(X_test)

        # Define the indices for the inner loop splits
        
        inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

        
        # Define the estimator and the parameter grid for GridSearchCV
        lr = LogisticRegression(class_weight='balanced',max_iter=1000)
        param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100]}

        clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
        
        # Fit the estimator on the training/validation set and calibrate the predictions
        clf.fit(X_train_val, y_train_val)
      

        # find the best cutoff
        
        #y_pred = calibrated_clf.predict(X_test)
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


    with mlflow.start_run(run_name=str(num_run+1),nested=True):   

        # Log the features used
        mlflow.log_param("ranking", num_run+1)
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
    

for i in tqdm(range(len(top_50_features[44:]))):
    features = top_50_features[:i+45]
    features = features + ['age_months','sex']
    logistic_regression_nested_cv(features,df,y,num_run=i+44)

logistic_regression_nested_cv(['age_months','sex'],df,y,num_run=-1)





mlflow.set_experiment("Tree CV IPW EO")



##################################
# Decision Tree  ####
###################################
def dt_nested_cv(feature_list,X,y,num_run):

    X = X.copy(deep=True)[feature_list]
    
    numeric_cols = [col for col in feature_list if col not in [ 'sex']]
    categorical_cols = [ 'sex']

    # Initialize the array to store the nested scores
    nested_sensitivity = np.zeros(10)
    nested_specificity = np.zeros(10)
    nested_roc_auc = np.zeros(10)
    nested_f1_macro = np.zeros(10)
    nested_accuracy = np.zeros(10)

    # Define the indices for the outer loop splits
    outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    outer_cv_splits = outer_cv.split(X, y)

    # Loop over the outer loop splits
    for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):
       
        # Split the data into training/validation and testing sets
        X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
        X_test, y_test = X.loc[test_idx,:], y[test_idx]
        

        preprocessor = ColumnTransformer(
        transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
        ])

        steps = [('preprocessor',preprocessor), ('impute', KNNImputer(n_neighbors=3))]
        pipeline = Pipeline(steps=steps)

        # Apply the pipeline on the training/validation set
        X_train_val = pipeline.fit_transform(X_train_val)

        # Apply the pipeline on the testing set
        X_test = pipeline.fit_transform(X_test)

        # Define the indices for the inner loop splits
     
        inner_cv =  StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

        
        # Define the estimator and the parameter grid for GridSearchCV
        dt = DecisionTreeClassifier(random_state=7)
        param_grid = {'max_depth': [3, 4, 5, 6, 7],
                      'min_samples_split': [2, 4, 6],
                      'min_samples_leaf': [1, 2, 4],
                      'max_features': ['sqrt']}

        clf = GridSearchCV(estimator=dt, param_grid=param_grid, cv=inner_cv,n_jobs = -1)

        
        # Fit the estimator on the training/validation set and calibrate the predictions
        clf.fit(X_train_val, y_train_val)
      

        # find the best cutoff
        
        #y_pred = calibrated_clf.predict(X_test)
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


    with mlflow.start_run(run_name=str(num_run+1),nested=True):   

        # Log the features used
        mlflow.log_param("ranking", num_run+1)
       

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
    

for i in tqdm(range(len(top_50_features))):
    features = top_50_features[:i+1]
    features = features + ['age_months','sex']
    dt_nested_cv(features,df,y,num_run=i)





mlflow.set_experiment("Random Forest Nested CV FULL MATCH EO")

##################################
# Random Forest##
###################################
def random_forest_nested_cv(feature_list,X,y,num_run):

    X = X.copy(deep=True)[feature_list]
    
    numeric_cols = [col for col in feature_list if col not in [ 'sex']]
    categorical_cols = ['sex']

    # Initialize the array to store the nested scores
    nested_sensitivity = np.zeros(10)
    nested_specificity = np.zeros(10)
    nested_roc_auc = np.zeros(10)
    nested_f1_macro = np.zeros(10)
    nested_accuracy = np.zeros(10)

    # Define the indices for the outer loop splits
    outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    outer_cv_splits = outer_cv.split(X, y)

    # Loop over the outer loop splits
    for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):
       
        # Split the data into training/validation and testing sets
        X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
        X_test, y_test = X.loc[test_idx,:], y[test_idx]
        

        preprocessor = ColumnTransformer(
        transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
        ])

        steps = [('preprocessor',preprocessor), ('impute', KNNImputer(n_neighbors=3))]
        pipeline = Pipeline(steps=steps)

        # Apply the pipeline on the training/validation set
        X_train_val = pipeline.fit_transform(X_train_val)

        # Apply the pipeline on the testing set
        X_test = pipeline.fit_transform(X_test)

        # Define the indices for the inner loop splits
     
        inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

        
        # Define the estimator and the parameter grid for GridSearchCV
        rf_params = {
        'n_estimators': [30,50,100],
        'max_depth': [3,None],
        'min_samples_split': [2,4],
        'min_samples_leaf': [1,4]}

        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

        clf = GridSearchCV(estimator=rf, param_grid=rf_params, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
        
        # Fit the estimator on the training/validation set and calibrate the predictions
        clf.fit(X_train_val, y_train_val)
        #calibrated_clf = CalibratedClassifierCV(clf,  cv='prefit')
        #calibrated_clf.fit(X_train_val, y_train_val)
        
        #y_pred = calibrated_clf.predict(X_test)
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


    with mlflow.start_run(run_name=str(num_run+1),nested=True):   

        # Log the features used
        mlflow.log_param("ranking", num_run+1)
        mlflow.log_param("n_estimators", clf.best_params_['n_estimators'])
        mlflow.log_param("max_depth", clf.best_params_['max_depth'])
        mlflow.log_param("min_samples_split", clf.best_params_['min_samples_split'])
        mlflow.log_param("min_samples_leaf", clf.best_params_['min_samples_leaf'])
      
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
    

for i in tqdm(range(len(top_50_features))):
    features = top_50_features[:i+1]
    features = features + ['age_months','sex']
    random_forest_nested_cv(features,df,y,num_run=i)

random_forest_nested_cv(['age_months','EOEC','sex'],df,y,num_run=-1)


#################################
### Elastic Net
#####################################


num_run = 1046
X = df
feature_list = full_features+['age_months', 'sex']
X = X.copy(deep=True)[feature_list]

numeric_cols = [col for col in feature_list if col not in ['sex','alpha_presence']]
categorical_cols = [ 'sex','alpha_presence']

# Initialize the array to store the nested scores
nested_sensitivity = np.zeros(10)
nested_specificity = np.zeros(10)
nested_roc_auc = np.zeros(10)
nested_f1_macro = np.zeros(10)
nested_accuracy = np.zeros(10)

# Define the indices for the outer loop splits
outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
outer_cv_splits = outer_cv.split(X, y)

# Loop over the outer loop splits
for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):
    
    # Split the data into training/validation and testing sets
    X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
    X_test, y_test = X.loc[test_idx,:], y[test_idx]
    

    preprocessor = ColumnTransformer(
    transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
    ])

    steps = [('preprocessor',preprocessor), ('impute', KNNImputer(n_neighbors=3))]
    pipeline = Pipeline(steps=steps)

    # Apply the pipeline on the training/validation set
    X_train_val = pipeline.fit_transform(X_train_val)

    # Apply the pipeline on the testing set
    X_test = pipeline.fit_transform(X_test)

    # Define the indices for the inner loop splits
    
    inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

    
    # Define the estimator and the parameter grid for GridSearchCV
        # Define the estimator and the parameter grid for GridSearchCV
    param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'l1_ratio': [0, 0.5, 1],
    'max_iter': [1000]}

# Create an SGDClassifier object
    elasticnet = SGDClassifier(loss='log_loss', penalty='elasticnet')

    clf = GridSearchCV(estimator=elasticnet, param_grid=param_grid, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
    
    # Fit the estimator on the training/validation set and calibrate the predictions
    clf.fit(X_train_val, y_train_val)
    #calibrated_clf = CalibratedClassifierCV(clf,  cv=5)
    #calibrated_clf.fit(X_train_val, y_train_val)
    
    #y_pred = calibrated_clf.predict(X_test)
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


with mlflow.start_run(run_name=str(num_run+1),nested=True):   

    # Log the features used
    mlflow.log_param("ranking", num_run+1)
    mlflow.log_param("l1_ratio", clf.best_params_['l1_ratio'])
    mlflow.log_param("alpha", clf.best_params_['alpha'])

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

















mlflow.set_experiment("SVM Nested CV 2")

##################################
# Support Vector Machine##
###################################
def svm_nested_cv(feature_list,X,y,num_run):

    X = X.copy(deep=True)[feature_list]
    
    numeric_cols = [col for col in feature_list if col not in ['EOEC', 'sex']]
    categorical_cols = ['EOEC', 'sex']

    # Initialize the array to store the nested scores
    nested_sensitivity = np.zeros(10)
    nested_specificity = np.zeros(10)
    nested_roc_auc = np.zeros(10)
    nested_f1_macro = np.zeros(10)
    nested_accuracy = np.zeros(10)

    # Define the indices for the outer loop splits
    outer_cv = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    outer_cv_splits = outer_cv.split(X, y)

    # Loop over the outer loop splits
    for j, (train_val_idx, test_idx) in enumerate(outer_cv_splits):
       
        # Split the data into training/validation and testing sets
        X_train_val, y_train_val = X.loc[train_val_idx,:], y[train_val_idx]
        X_test, y_test = X.loc[test_idx,:], y[test_idx]
        

        preprocessor = ColumnTransformer(
        transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
        ])

        steps = [('preprocessor',preprocessor), ('impute', KNNImputer(n_neighbors=3))]
        pipeline = Pipeline(steps=steps)

        # Apply the pipeline on the training/validation set
        X_train_val = pipeline.fit_transform(X_train_val)

        # Apply the pipeline on the testing set
        X_test = pipeline.transform(X_test)

        # Define the indices for the inner loop splits
     
        inner_cv =  RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=3)

        
        # Define the estimator and the parameter grid for GridSearchCV
        svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']}



        svm = SVC(class_weight='balanced', probability=True, random_state=42)

        clf = GridSearchCV(estimator=svm, param_grid=svm_params, cv=inner_cv,n_jobs=-1,scoring='roc_auc')
        
        # Fit the estimator on the training/validation set and calibrate the predictions
        clf.fit(X_train_val, y_train_val)
        #calibrated_clf = CalibratedClassifierCV(clf,  cv='prefit')
        #calibrated_clf.fit(X_train_val, y_train_val)
        
        #y_pred = calibrated_clf.predict(X_test)
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


    with mlflow.start_run(run_name=str(num_run+1),nested=True):   

        # Log the features used
        mlflow.log_param("ranking", num_run+1)
        mlflow.log_param("kernel", clf.best_params_['kernel'])
        mlflow.log_param("C", clf.best_params_['C'])
        mlflow.log_param("gamma", clf.best_params_['gamma'])

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
    

for i in tqdm(range(len(top_50_features[:10]))):
    features = top_50_features[:i+1]
    features = features + ['age_months','EOEC','sex']
    svm_nested_cv(features,df,y,num_run=i)

svm_nested_cv(['age_months','EOEC','sex'],df,y,num_run=-1)


##################################
# LightGBM #######################
##################################

mlflow.set_experiment("LightGBM Nested CV")

def lightgbm_nested_cv(feature_list, X, y, num_run):
    cv_inner = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=7)
    
    X = X.copy(deep=True)[feature_list]
    
    # Define the hyperparameter grid to search
    numeric_cols = [col for col in feature_list if col not in ['EOEC', 'sex']]
    categorical_cols = ['EOEC', 'sex']


    ########## INNER CV ##########
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

    steps = [('preprocessor',preprocessor),
             ('model', lightgbm.LGBMClassifier(n_jobs=-1,random_state=42,is_unbalance=True))]

    pipeline = Pipeline(steps=steps)

    param_grid = {
              'model__max_depth': [3,5,-1],
                'model__num_leaves': [10, 20, 31],
                'model__n_estimators': [30,40,50,70]
                }

  

    # Define the RandomizedSearchCV object
    lgbm_random_search = GridSearchCV(pipeline,
    param_grid=param_grid, scoring='roc_auc', cv=cv_inner, n_jobs=-1)
    # Fit the RandomizedSearchCV object to the data
    lgbm_random_search.fit(X, y)

    lgbm_random_search.best_params_
    # Get the best hyperparameters
    best_max_depth = lgbm_random_search.best_params_['model__max_depth']
    best_num_leaves = lgbm_random_search.best_params_['model__num_leaves']
    best_n_estimators = lgbm_random_search.best_params_['model__n_estimators']
    #best_pos_bagging_fraction = lgbm_random_search.best_params_['model__pos_bagging_fraction'],
    #best_neg_bagging_fraction = lgbm_random_search.best_params_['model__neg_bagging_fraction']

    # Define the LGBMClassifier object with the best hyperparameters
    best_lgbm = lightgbm.LGBMClassifier(n_jobs=-1, 
                                        max_depth=best_max_depth, 
                                        #pos_bagging_fraction = best_pos_bagging_fraction,
                                        #neg_bagging_fraction = best_neg_bagging_fraction,
                                        num_leaves=best_num_leaves,
                                        n_estimators=best_n_estimators,
                                        is_unbalance=True,
                                        random_state=42)

    # Define the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', best_lgbm)])

    # Run cross_validate with the pipeline
    scoring = {
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(recall_score, pos_label=0),
        'roc_auc': 'roc_auc',
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy'
    }

    # Start MLflow run
    with mlflow.start_run(run_name=str(num_run + 1), nested=True):
        # Log the features used
        mlflow.log_param("ranking", num_run + 1)
        best_params = lgbm_random_search.best_params_
        mlflow.log_params(best_params)

        scores_lgbm = cross_validate(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv_outer,
            n_jobs=-1,
            return_train_score=False
        )

        # Log each metric with its corresponding value
        for metric in scoring:
            test_scores = scores_lgbm['test_' + metric]
            best_score = test_scores.mean()
            mlflow.log_metric(metric, best_score)

    # End the MLflow run
    mlflow.end_run()


for i in tqdm(range(len(top_50_features))):
    features = top_50_features[:i+1]
    features = features + ['age_months','EOEC','sex']
    lightgbm_nested_cv(features,df,y,num_run=i)


###################################
### Random Forest #############
################################

def random_forest_nested_cv(feature_list, X, y, num_run):
    cv_inner = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=7)

    X = X.copy(deep=True)[feature_list]

    # Define the hyperparameter grid to search
    numeric_cols = [col for col in feature_list if col not in ['EOEC', 'sex']]
    categorical_cols = ['EOEC', 'sex']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    rf_params = {
        'model__n_estimators': [30,50,100],
        'model__max_depth': [3,None],
        'model__min_samples_split': [2,4],
        'model__min_samples_leaf': [1,4],
        'model__class_weight': ['balanced']
    }

    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('impute', SimpleImputer(strategy='mean')),('model', rf)])

    rf_random_search = GridSearchCV(pipeline, param_grid=rf_params, scoring='roc_auc', cv=cv_inner, n_jobs=-1)
    rf_random_search.fit(X, y)

    # Get the best hyperparameters
    best_params = rf_random_search.best_params_

    best_max_depth = rf_random_search.best_params_['model__max_depth']
    best_n_estimators = rf_random_search.best_params_['model__n_estimators']
    best_min_samples_split = rf_random_search.best_params_['model__min_samples_split']
    best_min_samples_leaf = rf_random_search.best_params_['model__min_samples_leaf']
    best_rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, 
                                     max_depth=best_max_depth,
                                     n_estimators=best_n_estimators,
                                     min_samples_split=best_min_samples_split,
                                     min_samples_leaf=best_min_samples_leaf)


    # Define the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('impute', SimpleImputer(strategy='mean')),('model', best_rf)])

    # Run cross_validate with the pipeline
    scoring = {
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(recall_score, pos_label=0),
        'roc_auc': 'roc_auc',
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy'
    }

    # Start MLflow run
    with mlflow.start_run(run_name=str(num_run+1), nested=True):
        # Log the features used
        mlflow.log_param("ranking", num_run+1)
        mlflow.log_params(best_params)

        scores_rf = cross_validate(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv_outer,
            n_jobs=-1,
            return_train_score=False
        )

        # Log each metric with its corresponding value
        for metric in scoring:
            test_scores = scores_rf['test_' + metric]
            best_score = test_scores.mean()
            mlflow.log_metric(metric, best_score)

    # End the MLflow run
    mlflow.end_run()




mlflow.set_experiment("RF Nested CV IPW")



for i in tqdm(range(len(top_50_features))):
    features = top_50_features[:i+1]
    features = features + ['age_months','EOEC','sex']
    random_forest_nested_cv(features,df,y,num_run=i)


random_forest_nested_cv(['age_months','EOEC','sex'],df,y,num_run=0)










