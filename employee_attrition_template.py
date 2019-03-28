# -*- coding: utf-8 -*-
"""
This file was made in Spyder Editor

Created on Sat Mar 17 23:31:45 2019

@author: jevon
"""
# Template built using https://towardsdatascience.com/building-an-employee-churn-model-in-python-to-develop-a-strategic-retention-plan-57d5bd882c2d?gi=a875e936cad7

# GitHub repo: https://github.com/hamzaben86/Employee-Churn-Predictive-Model

#Note: Just a skeleton and therefore missing sections.
#Can be updated later.

#This is a supervised classification problem.

import numpy as np
from openpyxl import load_workbook
from scipy import stats
from scipy.stats import norm, skew
import statsmodels.api as sm
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile

#data visualisation
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib 
# % matplotlib inline
color = sns.color_palette()
from IPython.display import display
pd.options.display.max_columns = None

#plotly
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

#plotly offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf #Need to research this.
cf.set_config_file(offline=True)
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#ML model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#data modeling
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier #Need to research this.
from sklearn.ensemble import RandomForestClassifier

#helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score

#misc
import os
import re
import sys
import timeit
import string
from datetime import datetime
from time import time
from dateutil.parser import parse
 
df_source = pd.read_excel('', sheet_name = 0) #add path to Excel source file
print("Shape of dataframe is: {}".format(df_source.shape))

df_human_resources = df_source.copy()

df_human_resources.columns

df_human_resources.head()

df_human_resources.columns.to_series().groupby(df_human_resources.dtypes).groups

#Datatypes and missing values
df_human_resources.info()

#Overview of numerical features
df_human_resources.describe()

df_human_resources.hist(figsize=(25,25))
plt.show()

#Overview of features by attribute

#Begin Age data
(mu, sigma) = norm.fit(df_human_resources.loc[df_human_resources['Attrition'] == 'Yes', 'Age'])
print('Ex: average age = {:0.2f} years with standard deviation = {0.2f}' .format(mu, sigma))
(mu, sigma) = norm.fit(df_human_resources.loc[df_human_resources['Attrition'] == 'No', 'Age'])
print('Current: average age = {:0.2f} years with standard deviation = {0.2f}' .format(mu, sigma))

x1 = df_human_resources.loc[df_human_resources['Attrition'] == 'No', 'Age']
x2 = df_human_resources.loc[df_human_resources['Attrition'] == 'Yes', 'Age']

hist_data = [x1, x2]
group_labels = ['Active', 'Inactive']

fig = ff.create_distplot(hist_data, group_labels, curve_type = 'kde', show_hist = False, show_rug = False)

fig['Layout'].update(title = 'Age Distritbuion by Attrition')
fig['Layout'].update(xaxis = dict(range=[10, 60], dticks = 5))

py.iplot(fig, filename = 'Distplot with Multiple Datasets')

#Educational Background areas

df_human_resources['EducationField'].value_counts()


df_EducationField = pd.DataFrame(columns=["EducationField", "% of Leavers"])
i = 0
for field in list(df_human_resources['EducationField'].unique()):
    ratio = (df_human_resources[(df_human_resources['EducationField']==field)&(df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources["EducationField"]==field].shape[0])
    df_EducationField.loc[i] = (field, ratio*100)
    i += 1

df_EducationFieldGroup = df_EducationField.groupby(by="EducationField").sum()
df_EducationFieldGroup.iplot(kind='bar', title='Leavers by Education field (%)')

#Gender distribution

df_human_resources['Gender'].value_counts()

print("Normalised gender distribution of ex-employees in the dataset: Male = {:0.2f}%; Female = {:0.2f}%.".format((df_human_resources[(df_human_resources['Attrition']=="Yes") & (df_human_resources['Gender'] == 'Male')].shape[0] / df_human_resources[df_human_resources['Gender']=='Male'].shape[0])*100, (df_human_resources[(df_human_resources['Attrition']=="Yes") & (df_human_resources['Gender'] == 'Female')].shape[0] / df_human_resources[df_human_resources['Gender']=='Female'].shape[0])*100))

df_Gender = pd.DataFrame(columns=["Gender", "% of Leavers"])
i = 0
for field in list(df_human_resources['Gender'].unique()):
    ratio = df_human_resources[(df_human_resources['Gender']==field) & df_human_resources(['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['Gender'] == field].shape[0]
    i += 1

df_GenderGroup = df_Gender.groupby(by = "Gender").sum()
df_GenderGroup.iplot(kind = 'bar', title = 'Leavers by Gender (%)')

#Marital Status

df_human_resources['MartalStatus'].value_counts()

df_MaritalStatus = pd.DataFrame(columns=["MaritalStatus", "% of Leavers"])
i=0
for field in list(df_human_resources['MaritalStatus'].unique()):
    ratio = df_human_resources[(df_human_resources['MaritalStatus']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['MaritalStatus']==field].shape[0]
    df_MaritalStatus.loc[i] = (field, ratio*100)
    i += 1

df_MaritalStatusGroup = df_MaritalStatus.groupby(by="MaritalStatus").sum()
df_MaritalStatusGroup.iplot(kind='bar', title='Leavers by Marital Status (%)')



#Distance from Home

print("Distance from home for employees to get to work is from {:0.2f} to {:0.2f} miles.".format(df_human_resources['DistanceFromHome'].min(), df_human_resources['DistanceFromHome'].max()))

print('Average distance from home for currently active employees: {:0.2f} miles and ex-employees: {:0.2f} miles'.format(df_human_resources[df_human_resources['Attrition']=='No']['DistanceFromHome'].mean(), df_human_resources[df_human_resources['Attrition']=='Yes']['DistanceFromHome'].mean()))

x1 = df_human_resources.loc[df_human_resources['Attrition']=='No', 'DistanceFromHome']
x2 = df_human_resources.loc[df_human_resources['Attrition']=='Yes', 'DistanceFromHome']

hist_data = [x1, x2]
group_labels = ['Active Employees', 'Non-Active Amployees']

fig = ff.create_distplot(hist_data, group_labels, curve_type='kde', show_hist=False, show_rug=False)

fig['layout'].update(title='Distance from Home Distribution in Percent by Status')
fig['layout'].update(xaxis=dict(range=[0,50], dtick=5))

py.iplot(fig, filename='Distplot with Multiple Datasets')

#Begin Department data analysis

df_human_resources['Department'].value_counts()

df_Department = pd.DataFrame(columns = ["Department", "% of Leavers"])
i=0
for field in list(df_human_resources(df_human_resources['Department'].unique())):
    ratio = df_human_resources[(df_human_resources['Department']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['Department']==field].shape[0]
    df_Department.loc[i] = (field, ratio*100)
    i += 1

df_DepartmentGroup = df_human_resources['Department'].groupby(by="Department").sum()
df_DepartmentGroup.iplot(kind='bar', title="Leavers by Department (%)")

#Frequency of Travel, Job Roles, Job Level, and Job Involvement

df_human_resources['BusinessTravel'].value_counts()

df_BusinessTravel = pd.DataFrame(columns=["BusinessTravel", "% of Leavers"])
i = 0
for field in list(df_human_resources['BusinessTravel'].unique()):
    ratio = df_human_resources[(df_human_resources['BusinessTravel']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['BusinessTravel']==field].shape[0]
    df_BusinessTravel.loc[i] = (field, ratio*100)
    i += 1

df_BusinessTravel_Group = df_BusinessTravel.groupby(by="BusinessTravel").sum()
df_BusinessTravel_Group.iplot(kind='bar', title='Leavers by Business Travel (%)')

df_human_resources['JobRole'].value_counts()

df_JobRole = pd.DataFrame(columns=["JobRole", "% of Leavers"])
i = 0
for field in list(df_human_resources['JobRole'].unique()):
    ratio = df_human_resources[(df_human_resources['JobRole']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['JobRole']==field].shape[0]
    df_JobRole.loc[i] = (field, ratio*100)
    i += 1

df_JobRole_Group = df_JobRole.groupby(by="JobRole").sum()
df_JobRole_Group.iplot(kind='bar', title='Leavers by Job Role (%)')

df_human_resources['JobLevel'].value_counts()

df_JobLevel = pd.DataFrame(columns=["JobLevel", "% of Leavers"])
i = 0
for field in list(df_human_resources['JobLevel'].unique()):
    ratio = df_human_resources[(df_human_resources['JobLevel']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['JobLevel']==field].shape[0]
    df_JobLevel.loc[i] = (field, ratio*100)
    i += 1

df_JobLevel_Group = df_JobLevel.groupby(by="JobLevel").sum()
df_JobLevel_Group.iplot(kind='bar', title='Leavers by Job Level (%)')

df_human_resources['JobInvolvement'].value_counts()

df_JobInvolvement = pd.DataFrame(columns=["JobInvolvement", "% of Leavers"])
i = 0
for field in list(df_human_resources['JobInvolvement'].unique()):
    ratio = df_human_resources[(df_human_resources['JobInvolvement']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['JobInvolvement']==field].shape[0]
    df_JobInvolvement.loc[i] = (field, ratio*100)
    i += 1

df_JobInvolvement_Group = df_JobInvolvement.groupby(by="JobInvolvement").sum()
df_JobInvolvement_Group.iplot(kind='bar', title='Leavers by Job Involvement (%)')

#Training incidents

print("Number of training incidents last year varies from {:0.2f} to {:0.2f} years.".format(df_human_resources['TrainingTimesLastYear'].min(), df_human_resources['TrainingTimesLastYear'].max()))

x1 = df_human_resources.loc[df_human_resources['Attrition'] == 'No', 'TrainingTimesLastYear']
x2 = df_human_resources.loc[df_human_resources['Attrition'] == 'Yes', 'TrainingTimesLastYear']

hist_data = [x1, x2]
group_labels = ['Active', 'Inactive']

fig = ff.create_distplot(hist_data, group_labels, curve_type = 'kde', show_hist = False, show_rug = False)

fig['Layout'].update(title = 'Distritbuion of Training Times Last Year by Attrition')
fig['Layout'].update(xaxis = dict(range=[10, 60], dticks = 5))

py.iplot(fig, filename = 'Distplot with Multiple Datasets')


#Number of Companies worked prior

df_NumberOfCompaniesWorked = pd.DataFrame(columns=["NumCompaniesWorked", "% of Leavers"])
i = 0
for field in list(df_human_resources['NumCompaniesWorked'].unique()):
    ratio = df_human_resources[(df_human_resources['NumCompaniesWorked']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['NumCompaniesWorked']==field].shape[0]
    df_JobLevel.loc[i] = (field, ratio*100)
    i += 1

df_NumberOfCompaniesWorked_Group = df_NumberOfCompaniesWorked.groupby(by="NumCompaniesWorked").sum()
df_NumberOfCompaniesWorked_Group.iplot(kind='bar', title='Leavers by Number of Prior Companies Worked (%)')


#Number of Years at Company

df_human_resources

print("The number of years spent at this company varies from  {:0.2f} to {:0.2f}.".format(df_human_resources['YearsAtCompany'].min(), df_human_resources['YearsAtCompany'].max()))

print('Average number of years spent at the company for currently active employees: {:0.2f}, and ex-employees: {:0.2f}'.format(df_human_resources[df_human_resources['Attrition']=='No']['YearsAtCompany'].mean(), df_human_resources[df_human_resources['Attrition']=='Yes']['YearsAtCompany'].mean()))

x1 = df_human_resources.loc[df_human_resources['Attrition']=='No', 'YearsAtCompany']
x2 = df_human_resources.loc[df_human_resources['Attrition']=='Yes', 'YearsAtCompany']

hist_data = [x1, x2]
group_labels = ['Active Employees', 'Non-Active Amployees']

fig = ff.create_distplot(hist_data, group_labels, curve_type='kde', show_hist=False, show_rug=False)

fig['layout'].update(title='Years at Company Distribution in Percent by Status')
fig['layout'].update(xaxis=dict(range=[0,50], dtick=5))

py.iplot(fig, filename='Distplot with Multiple Datasets')


#Years with Current Manager

print("The number of years spent with the current manager varies from  {:0.2f} to {:0.2f}.".format(df_human_resources['YearsWithCurrManager'].min(), df_human_resources['YearsWithCurrManager'].max()))

print('Average number of years spent with the current manager for currently active employees: {:0.2f}, and ex-employees: {:0.2f}'.format(df_human_resources[df_human_resources['Attrition']=='No']['YearsWithCurrManager'].mean(), df_human_resources[df_human_resources['Attrition']=='Yes']['YearsWithCurrManager'].mean()))

x1 = df_human_resources.loc[df_human_resources['Attrition']=='No', 'YearsWithCurrManager']
x2 = df_human_resources.loc[df_human_resources['Attrition']=='Yes', 'YearsWithCurrManager']

hist_data = [x1, x2]
group_labels = ['Active Employees', 'Non-Active Amployees']

fig = ff.create_distplot(hist_data, group_labels, curve_type='kde', show_hist=False, show_rug=False)

fig['layout'].update(title='Years at Company Distribution in Percent by Status')
fig['layout'].update(xaxis=dict(range=[0,50], dtick=5))

py.iplot(fig, filename='Distplot with Multiple Datasets')


#Work Life Balance

df_human_resources['WorkLifeBalance'].value_counts()

df_WorkLifeBalance = pd.DataFrame(columns=["WorkLifeBalance", "% of Leavers"])
i = 0
for field in list(df_human_resources['WorkLifeBalance'].unique()):
    ratio = df_human_resources[(df_human_resources['WorkLifeBalance']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['WorkLifeBalance']==field].shape[0]
    df_WorkLifeBalance.loc[i] = (field, ratio*100)
    i += 1

df_WorkLifeBalance_Group = df_WorkLifeBalance.groupby(by="WorkLifeBalance").sum()
df_WorkLifeBalance_Group.iplot(kind='bar', title='Leavers by Work Life Balance(%)')


df_human_resources['StandardHours'].value_counts()


df_human_resources['OverTIme'].value_counts()

df_OverTime = pd.DataFrame(columns=["OverTime", "% of Leavers"])
i = 0
for field in list(df_human_resources['OverTime'].unique()):
    ratio = df_human_resources[(df_human_resources['OverTime']==field) & (df_human_resources['Attrition']=="Yes")].shape[0] / df_human_resources[df_human_resources['OverTime']==field].shape[0]
    df_WorkLifeBalance.loc[i] = (field, ratio*100)
    i += 1

df_WorkLifeBalance_Group = df_WorkLifeBalance.groupby(by="OverTime").sum()
df_WorkLifeBalance_Group.iplot(kind='bar', title='Leavers by Over Time (%)')


#Compensation information

print("Employee Hourly Rate ranges from {:0.2f} to {:0.2f} years.".format(df_human_resources['HourlyRate'].min(), df_human_resources['HourlyRate'].max()))

print("Employee Daily Rate ranges from {:0.2f} to {:0.2f} years.".format(df_human_resources['DailyRate'].min(), df_human_resources['DailyRate'].max()))

print("Employee Monthly Rate ranges from {:0.2f} to {:0.2f} years.".format(df_human_resources['MonthlyRate'].min(), df_human_resources['MonthlyRate'].max()))

print("Employee Monthly Income ranges from {:0.2f} to {:0.2f} years.".format(df_human_resources['MonthlyIncome'].min(), df_human_resources['MonthlyIncome'].max()))

x1 = df_human_resources.loc[df_human_resources['Attrition'] == 'No', 'MonthlyIncome']
x2 = df_human_resources.loc[df_human_resources['Attrition'] == 'Yes', 'MonthlyIncome']

hist_data = [x1, x2]
group_labels = ['Active', 'Inactive']

fig = ff.create_distplot(hist_data, group_labels, curve_type = 'kde', show_hist = False, show_rug = False)

fig['Layout'].update(title = 'Distritbuion of Employee Monthly Income by Attrition')
fig['Layout'].update(xaxis = dict(range=[10, 60], dticks = 5))

py.iplot(fig, filename = 'Distplot with Multiple Datasets')


print("Percentage salary hikes range from {:0.2f} to {:0.2f} years.".format(df_human_resources['PercentSalaryHike'].min(), df_human_resources['PercentSalaryHike'].max()))

x1 = df_human_resources.loc[df_human_resources['Attrition'] == 'No', 'PercentSalaryHike']
x2 = df_human_resources.loc[df_human_resources['Attrition'] == 'Yes', 'PercentSalaryHike']

hist_data = [x1, x2]
group_labels = ['Active', 'Inactive']

fig = ff.create_distplot(hist_data, group_labels, curve_type = 'kde', show_hist = False, show_rug = False)

fig['Layout'].update(title = 'Distritbuion of Salary Hike Percents by Attrition')
fig['Layout'].update(xaxis = dict(range=[10, 60], dticks = 5))

py.iplot(fig, filename = 'Distplot with Multiple Datasets')



#Satisfaction and Performance




#Attrition and Correlation
df_human_resources['Attrition'].value_counts()

print("Percentage of Current Employees is {:.1f}% and of Ex-employees is: {:.1f}%".format(
    df_human_resources[df_human_resources['Attrition'] == 'No'].shape[0] / df_human_resources.shape[0]*100, df_human_resources[df_human_resources['Attrition'] == 'Yes'].shape[0] / df_human_resources.shape[0]*100))

df_human_resources['Attrition'].iplot(kind='hist', xTitle='Attrition', yTitle='count', title='Attrition Distribution')

df_human_resoures_transpose = df_human_resources.copy()
df_human_resoures_transpose['Target'] = df_human_resoures_transpose['Attrition'].apply(
    lambda x: 0 if x == 'No' else 1)
df_human_resoures_transpose = df_human_resoures_transpose.drop(
    ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)
correlations = df_human_resoures_transpose.corr()['Target'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))


corr = df_human_resoures_transpose.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr, vmax=.5, mask=mask, annot=True, fmt='0.2f', linewidths=.2, cmap="YlGnBu")

#label encoding object

label_encoder = LabelEncoder()

print(df_human_resources.shape)
df_human_resources.head()

#label encoding columns with values <= 2

label_encoder_count = 0
for col in df_human_resources.columns[1:] :
    if df_human_resources[col].dtype == 'object':
        if len(list(df_human_resources[col].unique())) <= 2 :
            label_encoder.fit(df_human_resources[col])
            df_human_resources[col] = label_encoder.transform(df_human_resources[col])
            label_encoder_count += 1

print('{} columns were label enconded.'.format(label_encoder_count))

df_human_resources = pd.get_dummies(df_human_resources, drop_first = True)

print(df_human_resources.shape)
df_human_resources.head()

scale = MinMaxScaler(feature_range=(0, 5))
humRes_col = list(df_human_resources.columns)
humRes_col.remove('Attrition')
for col in humRes_col:
    df_human_resources[col] = df_human_resources[col].astype(float)
    df_human_resources[[col]] = scale.fit_transform(df_human_resources[col])

df_human_resources['Attrition'] = pd.to_numeric(df_human_resources['Attrition'], downcast = 'float')
df_human_resources.head()

print('Size of fully enconded dataset: {}'.format(df_human_resources.shape))

#Assigning the target to a new dataframe and casting as a numerical feature
target = df_human_resources['Attrition'].copy()

trainX, testX, trainy, testy = train_test_split(df_human_resources, target, test_size = 0.25, random_state = 7, stratify = target)

print('Size of trainX dataset: ', trainX.shape)
print('Size of trainy dataset: ', trainy.shape)
print('Size of testX dataset: ', testX.shape)
print('Size of testy dataset: ', testy.shape)

#Logistic regression

models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier( n_estimators=100, random_state=7)))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=7)))
models.append(('Gaussian NB', GaussianNB()))

acc_results = []
auc_results = []
names = []
columns = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 
       'Accuracy Mean', 'Accuracy STD']
df_results = pd.DataFrame(columns=col)
i = 0
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cross_validation_acc_results = model_selection.cross_val_score(
        model, trainX, trainy, cross_validation=kfold, scoring='accuracy')
    cross_validation_auc_results = model_selection.cross_val_score(model, trainX, trainy, cross_validation=kfold, scoring='roc_auc')

    acc_results.append(cross_validation_acc_results)
    auc_results.append(cross_validation_auc_results)
    names.append(name)
    df_results.loc[i] = [name, round(cross_validation_acc_results.mean()*100, 2), round(cross_validation_auc_results.std()*100, 2), round(cross_validation_acc_results.mean()*100, 2), round(cross_validation_auc_results.std()*100, 2)]
    i += 1
df_results.sort_values(by=['ROC AUC Mean'], ascending=False)

fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc_results)
ax.set_xticklabels(names)
plt.show()

fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm ROC AUC Comparison')
ax = fig.add_subplot(111)
plt.boxplot(auc_results)
ax.set_xticklabels(names)
plt.show()

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression(solver='liblinear', class_weight="balanced", random_state=7)
scoring = 'roc_auc'
results = model_selection.cross_val_score(
    modelCV, trainX, trainy, cross_validation=kfold, scoring=scoring)
print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))


param_grid = {'alpha': np.arange(1e-03, 2, 0.01)} #hyperparameters
logis_gsch = GridSearchCV(LogisticRegression(solver = 'liblinear', class_weight = 'balanced', random_state = 7), iid = True, return_train_score = True, param_grid = param_grid, scoring = 'roc_auc', cross_validation = 10)

logis_grid = logis_gsch.fit(trainX, trainy)
logis_gopt = logis_grid.best_estimator_
result = logis_gsch.cv_results_

print('='*30)
print('best: ' + str(logis_gsch.best_estimator_))
print('best: ' + str(logis_gsch.best_params_))
print('best: ', logis_gsch.best_score_)
print('='*30)

kfold = model_selection.KFold(n_splits=10, random_state=7)
model_cross_validation = LogisticRegression(solver='liblinear', class_weight="balanced", random_state=7)
scoring = 'roc_auc'
results = model_selection.cross_val_score(model_cross_validation, trainX, trainy, cv=kfold, scoring=scoring)
print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))

logis_gopt.fit(trainX, trainy)
probably = logis_gopt.predict_proba(testX)
probably = probably[:, 1]
log_roc_auc = roc_auc_score(testy, probably)
print('AUC: %0.5f' % log_roc_auc)

#Random Forest

random_forest_classifier = RandomForestClassifier(class_weight = "balanced", random_state=7)
param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175], 'min_samples_split':[2,4,6,8,10], 'min_samples_leaf': [1, 2, 3, 4], 'max_depth': [5, 10, 15, 20, 25]}

grid_obj = GridSearchCV(random_forest_classifier, iid=True, return_train_score=True, param_grid=param_grid, scoring='roc_auc', cross_validation=10)

grid_fit = grid_obj.fit(trainX, trainy)
random_forest_optimization = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*20)


importances = random_forest_optimization.feature_importances_
indices = np.argsort(importances)[::-1]
names = [trainX.columns[i] for i in indices]
plt.figure(figsize=(15, 7))
plt.title("Feature Importance")
plt.bar(range(trainX.shape[1]), importances[indices])
plt.xticks(range(trainX.shape[1]), names, rotation=90)
plt.show()

importances = random_forest_optimization.feature_importances_
df_paramater_coefficient = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(44):
    feat = trainX.columns[i]
    coeff = importances[i]
    df_paramater_coefficient.loc[i] = (feat, coeff)
df_paramater_coefficient.sort_values(by='Coefficient', ascending=False, inplace=True)
df_paramater_coefficient = df_paramater_coefficient.reset_index(drop=True)
df_paramater_coefficient.head(10)

confuson_matrix = metrics.confusion_matrix(testy, random_forest_optimization.predict(testX))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confuson_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Accuracy of RandomForest Regression Classifier on test set: {:0.2f}'.format(random_forest_optimization.score(testX, testy)*100))
random_forest_optimization.fit(trainX, trainy)
print(classification_report(testy, random_forest_optimization.predict(testX)))

random_forest_optimization.fit(trainX, trainy)
probs = random_forest_optimization.predict_proba(testX)
probs = probs[:, 1]
random_forest_optimization_roc_auc = roc_auc_score(testy, probs)
print('AUC score: %0.3f' % random_forest_optimization_roc_auc)


ranFor_class = RandomForestClassifier(class_weight = 'balanced', random_state = 7)
param_grid = {'estimators': [50, 75, 100, 125, 150, 175, 200], 'min_samp_splt': [2, 4, 6, 8, 10], 'min_samp_lf': [1, 2, 3, 4, 5], 'max_depth': [5, 10, 15, 20, 25, 30]}
grid_object = GridSearchCV(ranFor_class, iid = True, return_train_score = True, param_grid = param_grid, scoring = 'roc_auc', cv = 10)

fitGrid = grid_object.fit(trainX, trainy)
ranFor_opt = fitGrid.best_estimator_

print('='*30)
print('best: ' + str(grid_object.best_estimator_))
print('best: ' + str(grid_object.best_params_))
print('best: ', grid_object.best_score_)
print('='*30)

ranFor_opt.fit(trainX, trainy)
probably2 = ranFor_opt.predict_proba(testX)
probably2 = probably2[:, 1]
ranFor_opt_roc_auc = roc_auc_score(testy, probably2)
print('AUC: %0.5f' % ranFor_opt_roc_auc)

fpr, tpr, thresholds = roc_curve(testy, logis_gopt.predict_proba(testX)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(testy, random_forest_optimization.predict_proba(testX)[:,1])
plt.figure(figsize=(14, 6))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % log_roc_auc)
# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % random_forest_optimization_roc_auc)
# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()