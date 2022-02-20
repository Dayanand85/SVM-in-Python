# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:51:38 2022

@author: Dayanand
"""
# loading library

import os
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.pyplot import figure

# setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# changing directory

os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\DataSets")

# loading datasets

fullRaw=pd.read_csv("BankCreditCard.csv")

#sampling data into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(fullRaw,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# adding source column in train & test

trainDf["Source"]="Train"
testDf["Source"]="Test"

# combining both datasets

fullDf=pd.concat([trainDf,testDf],axis=0)
fullDf.shape

fullDf.columns
fullDf.drop(["Customer ID"],axis=1,inplace=True)
fullDf_summary=fullDf.describe
fullDf_summary

# check NA values

fullDf.isna().sum() # no NA values

# chacking event rate

fullDf["Default_Payment"].value_counts()/fullDf.shape[0]
#77% & 22%

#############
####
## Use data description table to convert categorical numerical variable to proper categorical variables
##############

# Gender,Academic_Qualification,Narital
# Gender
var_to_update="Gender"
fullDf[var_to_update].value_counts()
fullDf[var_to_update]=fullDf[var_to_update].replace({1:"Male",2:"Female"}) 
fullDf[var_to_update].value_counts()

# Academic_Qualification

var_to_update="Academic_Qualification"
fullDf[var_to_update].value_counts()
fullDf[var_to_update].replace({1:"Undergraduate",
                               2:"Graduate",
                               3:"Postgraduate",
                               4:"Professional",
                               5:"Others",
                               6:"Unknown"},inplace=True)
fullDf[var_to_update].value_counts()

# Marital
var_to_update="Marital"
fullDf[var_to_update].value_counts()
fullDf[var_to_update].replace({1:"Married",
                               2:"Single",
                               3:"Do not prefer",
                               0:"Do not prefer"},inplace=True)
fullDf[var_to_update].value_counts()

# Combining Academic_Qualification columns into few categories
tempDf=fullDf[fullDf["Source"]=="Train"] 

# step 1-
propDf=pd.crosstab(tempDf["Academic_Qualification"],tempDf["Default_Payment"],margins=True)

# step 2-study the data for combining
propDf["Default_Col"]=round(propDf[1]/propDf["All"],1)

# step 3
fullDf["AQ_New"]=np.where(fullDf["Academic_Qualification"].isin(["Graduate","Undergraduate","Postgraduate"]),
                                                                  "Group1","Group2")
fullDf["AQ_New"].unique()
del fullDf["Academic_Qualification"]

# Dummy variable creation
fullRaw2=fullRaw.copy()
fullDf2=pd.get_dummies(fullDf,drop_first=False)
fullDf2.drop(["Source_Test"],axis=1,inplace=True)

#Divide the data into train & test
trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train"],axis=1)
testDf=fullDf2[fullDf2["Source_Train"]==0].drop(["Source_Train"],axis=1)

# Divide into Indep & Dependent columns
trainX=trainDf.drop(["Default_Payment"],axis=1)
trainY=trainDf["Default_Payment"]
testX=testDf.drop(["Default_Payment"],axis=1)
testY=testDf["Default_Payment"]
trainX.shape
testX.shape

#Model Building
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
# first Model
M1=SVC()
M1_Model=M1.fit(trainX,trainY)
test_class=M1_Model.predict(testX)
conf_mat=confusion_matrix(testY,test_class)
conf_mat
sum(np.diagonal(conf_mat))/testY.shape[0]*100
print(classification_report(testY,test_class))

# # 2nd Model
# M2=SVC(kernel="poly")
# M1_Model1=M1.fit(trainX,trainY)
# test_class1=M1_Model1.predict(testX)
# print(classification_report(testY,test_class1))

# # 3rd Model
# M3=SVC(kernel="linear")
# M1_Model2=M3.fit(trainX,trainY)
# test_class3=M1_Model2.predict(testX)
# print(classification_report(testY,test_class3))

# randomizedsearchCV

from sklearn.model_selection import RandomizedSearchCV
my_kernel=["sigmoid","rbf"]
my_cost=[0.1,1,2]
param_grid={"C":my_cost,"kernel":my_kernel}

SVM_RandomSearchCV=RandomizedSearchCV(SVC(),param_distributions=param_grid,cv=3,
                          scoring="f1",n_iter=3,n_jobs=-1,
                          random_state=2410).fit(trainX,trainY)

SVM_RandomSearchCV_Df=pd.DataFrame.from_dict(SVM_RandomSearchCV.cv_results_)

# Some moderate to advance level Data preparation steps

# Standardization

from sklearn.preprocessing import StandardScaler
Train_Scaling=StandardScaler().fit(trainX) # Inference step.Train_Scaling keep mean,variance
trainX_std=Train_Scaling.transform(trainX)
testX_std=Train_Scaling.transform(testX)

# Add column names
trainX_std=pd.DataFrame(trainX_std,columns=trainX.columns)
testX_std=pd.DataFrame(testX_std,columns=testX.columns)

# Model on standardized data
M2=SVC()
M2_Model=M2.fit(trainX_std,trainY)

Test_Predict2=M2_Model.predict(testX_std)
conf_mat2=confusion_matrix(testY,Test_Predict2)
conf_mat2
print(classification_report(testY,Test_Predict2))

# Startified Sampling-Class imbalance issue handling

from imblearn.under_sampling import RandomUnderSampler
RUS=RandomUnderSampler(sampling_strategy=0.7,random_state=2410)
trainX_RUS,trainY_RUS=RUS.fit_resample(trainX_std,trainY)
# trainX_RUS=pd.DataFrame(trainX_RUS)
# trainY_RUS=pd.Series(trainY_RUS)

# Model Building
M3=SVC()
Model_Build3=M3.fit(trainX_RUS,trainY_RUS)
Test_Predict4=Model_Build3.predict(testX_std)
conf_mat4=confusion_matrix(testY,Test_Predict4)
conf_mat4
print(classification_report(testY,Test_Predict4))

# Data Transformations-Data Transformations requires some mathematical function
import seaborn as sns
import numpy as np

colmnsToConsider=["Credit_Amount","Age_Years","Jan_Bill_Amount","Feb_Bill_Amount"]
trainX_copy=trainX.copy()

# pairplot histogram
sns.pairplot(trainX_copy[colmnsToConsider])

trainX_copy["Age_Years"]=np.log(np.where(trainX_copy["Age_Years"]==0,1,trainX_copy["Age_Years"]))
testX_copy=testX.copy()
testX_copy["Age_Years"]=np.log(np.where(testX_copy["Age_Years"]==0,1,testX_copy["Age_Years"]))
sns.pairplot(trainX_copy[colmnsToConsider])

sns.distplot(trainX_copy["Age_Years"])

# standardization
Train_Scaling=StandardScaler.fit(trainX_copy)
trainX_std=Train_Scaling.transform(trainX_copy)
testX_std=Train_Scaling.transform(testX_copy)

#Model Building
M5=SVC()
Model_Build5=M5.fit(trainX_std,trainY)
Test_Predict5=Model_Build5.predict(testX_std)
conf_mat4=confusion_matrix(testY,Test_Predict5)
print(classification_report(testY,Test_Predict5))
