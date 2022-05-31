# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:16:54 2022

@author: Dayanand
"""

# loading library

import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib.pyplot import figure


# setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

# changing the directory

os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\Imarticus hackathon\\Stage-2")

# loading file

rawDf=pd.read_csv("train_IA_-_train.csv")
predictionDf=pd.read_csv("test_IA_-_test.csv")


# Analysis of the datasets

rawDf.shape
predictionDf.shape # we have one column less
rawDf.columns
predictionDf.columns # We do not have status column
rawDf.head()

# Let us add status column

predictionDf["Status"]=np.nan

# Let us split the rawDf into train & test 

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# Let us add Source column in all three data sets to avoid data likeage

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

# Let us concat all three data sets

fullDf=pd.concat([trainDf,testDf,predictionDf],axis=0)
fullDf.shape

# Let us drop some identifier columns

fullDf.drop(['Interview Id', 'Candidate Id', 'Interviewer Id'],axis=1,inplace=True)
fullDf.shape

# Let us see the info of data sets

fullDf.info()

# Let us see the summary of the data sets

fullDf.describe().T

# Let us check the event rate of Output column

fullDf.loc[fullDf["Source"]=="Train","Status"].value_counts(normalize=True)

#fullDf.loc[fullDf["Source"]=="Train","Status"].value_counts()[0:1]
#fullDf.groupby("Status")["Status"].count()

# Let us change some columns values L.J.T.C & L.J.T.I columns

fullDf[["L.J.T.C","L.J.T.I"]]=fullDf[["L.J.T.C","L.J.T.I"]].replace(0,np.nan)

# # Let us check the missing values

fullDf.isna().sum()

# Missing value treatment

for i in fullDf.columns:
    if (i!="Source"):
        if fullDf[i].dtype=="object":
            tempMode=fullDf.loc[fullDf["Source"]=="Train",i].mode()[0]
            fullDf[i].fillna(tempMode,inplace=True)
        else:
            tempMed=fullDf.loc[fullDf["Source"]=="Train",i].median()
            fullDf[i].fillna(tempMed,inplace=True)

# Let us re confirm the missing values

fullDf.isna().sum()

# Let us do EDA

# Bivariate Analysis (For Categorical Indep vars) using Histogram

for i in fullDf.columns:
    if (i!="Status" and i!="Source"):
        if fullDf[i].dtype=="object":
            figure()
            sns.histplot(trainDf, x=i, hue="Status", stat="probability", multiple="fill")

# Observations-Developer & QA Manual profile has been bit more considered status while Marketing profile has bit less considereed
 
# Bivariate Analysis (For Continuous Vars) using Boxplot

trainDf = fullDf.loc[fullDf['Source'] == 'Train']
continuousVars = trainDf.columns[trainDf.dtypes != object]
continuousVars

for colNumber, colName in enumerate(continuousVars):
    figure()
    sns.boxplot(y = trainDf[colName], x = trainDf["Status"])
    trainDf.columns

##### Observation
# From Status (Consider) has high Interview Duration time comparatively others
# From Status (Consider) has low value for Q.A comparatively others
# From Status (Not Consider) has low value for S.P.C comparatively others
# From Status (Consider) has low value for L.J.T.C and (Not Consider) has very high value for L.J.T.C comparatively others

# Let us convert status columns 

fullDf["Status"].value_counts()
fullDf["Status"].replace({"Consider":1, 
                         "May Consider":2,
                         "Not Consider":3}, inplace = True)
fullDf["Status"].value_counts()

# dummy varibale creation

fullDf2=pd.get_dummies(fullDf)


# Divide the data sets 

Train=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
Train.shape
Test=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
Test.shape
Prediction=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
Prediction.shape

# Divide data sets into dep and indep var
trainX=Train.drop(["Status"],axis=1)
trainY=Train["Status"]

testX=Test.drop(["Status"],axis=1)
testY=Test["Status"]

predictionX=Prediction.drop(["Status"],axis=1)

##### Model Building

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# SVM Model (One Versus One)

svmModel_OVO = SVC(decision_function_shape="ovo").fit(trainX, trainY)

# Prediction

testPredSVM = svmModel_OVO.predict(testX)

# accuracy

print(accuracy_score(testY,testPredSVM))# 93%

#######
### GridSearch CV
#######

from sklearn.model_selection import GridSearchCV

# Setting parameter range

param_grid = {"C": [0.1, 1, 10, 100,1000],
              "kernel": ["rbf","poly"]}
 
grid_Search = GridSearchCV(SVC(decision_function_shape="ovo"), 
                           param_grid,scoring="accuracy",cv=3,n_jobs=-1)
 
# Model fit for grid search

GridSearchResult=grid_Search.fit(trainX, trainY)

# Best Score

GridSearchResult.best_score_ #0.97

# Best param parameters

GridSearchResult.best_params_

# Final Model Building Using hyper parameters

svmModel_OVO_Final = SVC(decision_function_shape="ovo", C = 100,random_state=2410).fit(trainX, trainY)

# Prediction

testPredSVMF = svmModel_OVO_Final.predict(testX)

# confusion matrix

pd.crosstab(testY,testPredSVMF)

# classification report

print(classification_report(testY,testPredSVMF))
    
# accuracy

print(accuracy_score(testY,testPredSVMF)) # 97%



# Prediction on Prediction Data Sets

OutputDf=pd.DataFrame()
OutputDf["Interview Id"]=predictionDf["Interview Id"]
OutputDf["Status"]=svmModel_OVO_Final.predict(predictionX)
OutputDf["Status"].replace({1:"Consider",
                            2:"May Consider",
                            3:"Not Consider"},inplace=True)
OutputDf.to_csv("sample_IA.csv",index=False)


