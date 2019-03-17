# -*- coding: utf-8 -*-
"""
This file was made in Spyder Editor
"""
# Template built using https://towardsdatascience.com/building-an-employee-churn-model-in-python-to-develop-a-strategic-retention-plan-57d5bd882c2d?gi=a875e936cad7

# GitHub repo: https://github.com/hamzaben86/Employee-Churn-Predictive-Model

#Note: Just a skeleton and therefore missing sections.
#Can be updated later.

import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
import numpy as np
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm

#data visualisation
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
%matplotlib inline
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
import cufflinks as cf
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
from xgboost import XGBClassifier
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

df_humRes = df_source.copy()

df_humRes.columns

df_humRes.head()

df_humRes.columns.to_series().groupby(df_humRes.dtypes).groups

#label encoding object

labEnc = LabelEncoder()

print(df_humRes.shape)
df_humRes.head()

#label encoding columns with values <= 2

labEnc_count = 0
for col in df_humRes.columns[1:] :
    if df_humRes[col].dtype == 'object':
        if len(list(df_humRes[col].unique())) <= 2 :
            labEnc.fit(df_humRes[col])
            df_humRes[col] = labEnc.transform(df_humRes[col])
            labEnc_count += 1

print('{} columns were label enconded.'.format(labEnc_count))

df_humRes = pd.get_dummies(df_humRes, drop_first = True)

print(df_humRes.shape)
df_humRes.head()

scale = MinMaxScaler(feature_range=(0, 5))
humRes_col = list(df_humRes.columns)
humRes_col.remove('Attrition')
for col in humRes_col:
    df_humRes[col] = df_humRes[col].astype(float)
    df_humRes[[col]] = scale.fit_transform(df_humRes[col])

df_humRes['Attrition'] = pd.to_numeric(df_humRes['Attrition'], downcast = 'float')
df_humRes.head()

print('Size of fully enconded dataset: {}'.format(df_humRes.shape))

#Assigning the target to a new dataframe and casting as a numerical feature
target = df_humRes['Attrition'].copy()

trainX, testX, trainy, testy = train_test_split(df_humRes, target, test_size = 0.25, random_state = 7, stratify = target)

print('Size of trainX dataset: ', trainX.shape)
print('Size of trainy dataset: ', trainy.shape)
print('Size of testX dataset: ', testX.shape)
print('Size of testy dataset: ', testy.shape)

#Logistic regression

#Add missing sections.
param_grid = {'alpha': np.arange(1e-03, 2, 0.01)} #hyperparameters
logis_gsch = GridSearchCV(LogisticRegression(solver = 'liblinear', class_weight = 'balanced', random_state = 7), iid = True, return_train_score = True, param_grid = param_grid, scoring = 'roc_auc', cv = 10)

logis_grid = logis_gsch,fit(trainX, trainy)
logis_gopt = logis_grid.best_estimator_
result = logis_gsch.cv_results_

print('='*30)
print('best: ' + str(logis_gsch.best_estimator_))
print('best: ' + str(logid_gsch.best_params_))
print('best: ', logis_gsch.best_score_)
print('='*30)

logis_gopt.fit(trainX, trainy) #fitting the model to the training data
probably = logis_gopt.predict_proba(testX) #predicting the probability
probably = probably[:, 1] #remove the probabilities where employee remained
log_roc_auc = roc_auc_score(testy, probably)
print('AUC: %0.5f' % log_roc_auc)

#Random Forest

#Add missing sections.
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

ranFor_opt.fit(trainX, trainy) #fit to the training data
probably2 = ranFor_opt.predict_proba(testX)
probably2 = probs[:, 1] #negate employee remaning
ranFor_opt_roc_auc = roc_auc_score(testy, probs)
print('AUC: %0.5f' % ranFor_opt_roc_auc)

#Add graph for ROC here.