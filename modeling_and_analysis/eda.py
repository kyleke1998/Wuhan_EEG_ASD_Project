import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import ast
import os
import cleaning
from cleaning import light_preprocessing, remove_bad_channels_kevin
from cleaning import all_batch_1_edfs, subjects_to_remove
import os
import seaborn as sns
from scipy import stats

pd.set_option('display.max_rows', None)


metadata = pd.read_csv('../eeg_data/wuhan_study_clinical_data.csv')
metadata['exposure'] = metadata['CARS_Categorical'].apply(lambda x: 'No ASD' if x == 'No ASD' else 'ASD')

#######################
## Spectral Power ##
########################

# plot power features for asd vs control
power_features = pd.read_feather('../features/all_power.feather')
power_names = [column for column in power_features.columns if column not in ['index','file_name','exp_group','subject']]
power_features = power_features.merge(metadata[['file_name','exposure','exp_group']],left_on='subject',right_on='file_name',how='inner')

power_features = power_features[power_features['exp_group'].isin(['ASD','No ASD'])]

power_diff = power_features.groupby('exposure')[power_names].mean().diff().abs()

# calculate average absolute difference
power_diff.iloc[1].sort_values(ascending=False)


ax = sns.violinplot(data=power_features,  x='bp_delta_Pz',y='exp_group', color="#af52f4", inner=None, linewidth=0, saturation=0.5)
sns.boxplot(data=power_features,  x='bp_delta_Pz', y='exp_group',saturation=0.5, width=0.4,
            palette='rocket', boxprops={'zorder': 2}, ax=ax)
sns.stripplot(data=power_features, x='bp_delta_Pz', y='exp_group', jitter=True, size=3, color='black', ax=ax)

plt.show()
plt.title('')

with PdfPages('../results/power_feature_distributions.pdf') as pdf:
    for column in power_names:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=power_features,  x=column,y='exposure', inner=None, linewidth=0, saturation=0.5)
        sns.boxplot(data=power_features,  x=column, y='exposure',saturation=0.5, width=0.4,
            palette='rocket', boxprops={'zorder': 2}, ax=ax)

        sns.stripplot(data=power_features, x=column, y='exposure', jitter=True, size=3, color='black', ax=ax)

        # Get the sample sizes for each group
        n_ASD = sum(~power_features[power_features['exposure'] == 'ASD'][column].isna())
        n_no_ASD = sum(~power_features[power_features['exposure'] == 'No ASD'][column].isna())
        
        # Add the sample sizes to the title
        plt.title('{} (n_ASD={}, n_no_ASD={})'.format(column, n_ASD, n_no_ASD))
        plt.xlabel('Relative_{}'.format(column))
        pdf.savefig(fig)
     

#######################
## Spectral Coherence ##
########################


coh_features = pd.read_feather('../features/all_spectral_coh.feather')
coh_names = [column for column in coh_features.columns if column not in ['index','file_name','subject','exp_group']]
coh_features = coh_features.merge(metadata[['file_name','exposure','exp_group']],left_on='subject',right_on='file_name')

coh_features = coh_features[coh_features['exp_group'].isin(['ASD','No ASD'])]


coh_features_diff = coh_features.groupby('exposure')[coh_names].mean().diff().abs()

coh_features_diff.iloc[1].sort_values(ascending=False)[:10]
# calculate average absolute difference
top_10_diff = coh_features_diff.iloc[1].sort_values(ascending=False).index[:10]


ax = sns.violinplot(data=coh_features,  x='gamma_coh_C4_O1',y='exp_group', color="#af52f4", inner=None, linewidth=0, saturation=0.5)
sns.boxplot(data=coh_features,  x='gamma_coh_C4_O1', y='exp_group',saturation=0.5, width=0.4,
            palette='rocket', boxprops={'zorder': 2}, ax=ax)
sns.stripplot(data=coh_features, x='gamma_coh_C4_O1', y='exp_group', jitter=True, size=3, color='black', ax=ax)

plt.show()


with PdfPages('../results/top_10_coh_diff.pdf') as pdf:
    for column in top_10_diff:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=coh_features,  x=column,y='exposure', inner=None, linewidth=0, saturation=0.5)
        sns.boxplot(data=coh_features,  x=column, y='exposure',saturation=0.5, width=0.4,
            palette='rocket', boxprops={'zorder': 2}, ax=ax)

        sns.stripplot(data=coh_features, x=column, y='exposure', jitter=True, size=3, color='black', ax=ax)

        # Get the sample sizes for each group
        n_ASD = sum(~coh_features[coh_features['exposure'] == 'ASD'][column].isna())
        n_no_ASD = sum(~coh_features[coh_features['exposure'] == 'No ASD'][column].isna())
        
        
        # Add the sample sizes to the title
        plt.title('{} (n_ASD={}, n_no_ASD={})'.format(column, n_ASD, n_no_ASD))
        plt.xlabel('{}'.format(column))
        pdf.savefig(fig)



#################
##sample entropy ##
################

sampen_features = pd.read_feather('../features/all_sampleEN.feather')
sampen_names = [column for column in sampen_features.columns if column not in ['index','file_name','subject','exp_group']]
sampen_features = sampen_features.merge(metadata[['file_name','exposure','exp_group']],left_on='subject',right_on='file_name')

sampen_features = sampen_features[sampen_features['exp_group'].isin(['ASD','No ASD'])]

sampen_features_diff = sampen_features.groupby('exposure')[sampen_names].mean().diff().abs()
sampen_features_diff.iloc[1].sort_values(ascending=False)
with PdfPages('../results/sampen_feature_distributions.pdf') as pdf:
    for column in sampen_names:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=sampen_features,  x=column,y='exposure', inner=None, linewidth=0, saturation=0.5)
        sns.boxplot(data=sampen_features,  x=column, y='exposure',saturation=0.5, width=0.4,
            palette='rocket', boxprops={'zorder': 2}, ax=ax)

        sns.stripplot(data=sampen_features, x=column, y='exposure', jitter=True, size=3, color='black', ax=ax)

        # Get the sample sizes for each group
        n_ASD = sum(~sampen_features[sampen_features['exposure'] == 'ASD'][column].isna())
        n_no_ASD = sum(~sampen_features[sampen_features['exposure'] == 'No ASD'][column].isna())
        
        
        # Add the sample sizes to the title
        plt.title('{} (n_ASD={}, n_no_ASD={})'.format(column, n_ASD, n_no_ASD))
        plt.xlabel('{}'.format(column))
        pdf.savefig(fig)



##############
# Mean features#
##############

mean_features = pd.read_feather('../features/all_statistical.feather')
mean_names = [column for column in mean_features.columns if column not in ['index','file_name','subject','exp_group']]
mean_features = mean_features.merge(metadata[['file_name','exposure','exp_group']],left_on='subject',right_on='file_name')

mean_features = mean_features[mean_features['exp_group'].isin(['ASD','No ASD'])]

mean_features_diff = mean_features.groupby('exposure')[mean_names].mean().diff().abs()
mean_features_diff.iloc[1].sort_values(ascending=False)



with PdfPages('../results/statistical_feature_distributions.pdf') as pdf:
    for column in mean_names:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=mean_features, x=column,y='exposure', inner=None, linewidth=0, saturation=0.5)
        sns.boxplot(data=mean_features, x=column, y='exposure',saturation=0.5, width=0.4,
        palette='rocket', boxprops={'zorder': 2}, ax=ax)

        sns.stripplot(data=mean_features, x=column, y='exposure', jitter=True, size=3, color='black', ax=ax)

        # Get the sample sizes for each group
        n_ASD = sum(~mean_features[mean_features['exp_group'] == 'exposure'][column].isna())
        n_no_ASD = sum(~mean_features[mean_features['exp_group'] == 'exposure'][column].isna())
        
        # Add the sample sizes to the title
        plt.title('{} (n_ASD={}, n_no_ASD={})'.format(column, n_ASD, n_no_ASD))
        plt.xlabel('{}'.format(column))
        pdf.savefig(fig)




alpha_features = pd.read_feather('../features/all_alpha.feather')
alpha_names = [column for column in mean_features.columns if column not in ['index','file_name','subject','exp_group']]
alpha_features = alpha_features.merge(metadata[['file_name','exposure','exp_group']],left_on='subject',right_on='file_name')

alpha_features = alpha_features[alpha_features['exp_group'].isin(['ASD','No ASD'])]


###########
# Concatenate all features into one feature matrix
###########


power_features = pd.read_feather('../features/all_power.feather')
power_features = power_features.merge(metadata[['file_name']],left_on='subject',right_on='file_name',how='inner')

coh_features = pd.read_feather('../features/all_spectral_coh.feather')
coh_features = coh_features.merge(metadata[['file_name']],left_on='subject',right_on='file_name')

sampen_features = pd.read_feather('../features/all_sampleEN.feather')
sampen_features = sampen_features.merge(metadata[['file_name']],left_on='subject',right_on='file_name')

alpha_features = pd.read_feather('../features/all_alpha.feather')
alpha_features = alpha_features.merge(metadata[['file_name']],left_on='subject',right_on='file_name')

mean_features = pd.read_feather('../features/all_statistical.feather')
mean_features = mean_features.merge(metadata[['file_name','exp_group','exposure','CARS_Categorical']],left_on='subject',right_on='file_name')



all_features = power_features.merge(coh_features,on=['file_name'],how='inner',suffixes=('_left', '_right'))
all_features = all_features.merge(sampen_features,on=['file_name'],how='inner',suffixes=('_left', '_right'))
all_features = all_features.merge(alpha_features,on=['file_name'],how='inner')
all_features = all_features.merge(mean_features,on=['file_name'],how='inner')

all_features = all_features[all_features['exp_group'].isin(['ASD','No ASD'])]
all_features  = all_features.drop(['index_left','index_right','subject_left','subject_right'], axis=1)


all_features.to_csv('../features/all_features.csv',index=False)





######
# plot top hits
######


top_hits = pd.read_csv('./top_fits_EC_exp_group_logit_aipw.csv')

top_hits_list = top_hits.outcome.tolist()

top_hits.columns

with PdfPages('../results/top_hits_exp_group_aipw.pdf') as pdf:
    for column in top_hits_list:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=all_features, x=column,y='exp_group', inner=None, linewidth=0, saturation=0.5)
        sns.boxplot(data=all_features, x=column, y='exp_group',saturation=0.5, width=0.4,
        palette='rocket', boxprops={'zorder': 2}, ax=ax)

        sns.stripplot(data=all_features, x=column, y='exp_group', 
                      jitter=True, size=3, hue='CARS_Categorical', palette='Set2', ax=ax)

        # Get the sample sizes for each group
        n_ASD = sum(~all_features[all_features['exp_group'] == 'ASD'][column].isna())
        n_no_ASD = sum(~all_features[all_features['exp_group'] == 'No ASD'][column].isna())
        
        # Add the sample sizes to the title
        plt.title('{} (n_ASD={}, n_no_ASD={}, BH_adjusted_p_value={})'.format(column, n_ASD, n_no_ASD,top_hits.query('outcome == @column')['fdr_p_value_aipw'].to_numpy()))
        plt.xlabel('{}'.format(column))
        pdf.savefig(fig)



with PdfPages('../results/top_hits_graphs_exp_group.pdf') as pdf:
    for column in top_hits_list:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=all_features, x=column,y='exposure', inner=None, linewidth=0, saturation=0.5)
        sns.boxplot(data=all_features, x=column, y='exposure',saturation=0.5, width=0.4,
        palette='rocket', boxprops={'zorder': 2}, ax=ax)

        sns.stripplot(data=all_features, x=column, y='exposure', 
                      jitter=True, size=3, hue='CARS_Categorical', palette='Set2', ax=ax)

        # Get the sample sizes for each group
        n_ASD = sum(~all_features[all_features['exposure'] == 'ASD'][column].isna())
        n_no_ASD = sum(~all_features[all_features['exposure'] == 'No ASD'][column].isna())
        
        # Add the sample sizes to the title
        plt.title('{} (n_ASD={}, n_no_ASD={})'.format(column, n_ASD, n_no_ASD))
        plt.xlabel('{}'.format(column))
        pdf.savefig(fig)


metadata[['exp_group','CARS_Categorical']].query('exp_group == "No ASD"')

metadata.query("exp_group == 'ASD' & CARS_Categorical == 'No ASD'")[['exp_group','Autism_Behavior_Checklist','CARS_Numerical','CARS_Categorical']]

