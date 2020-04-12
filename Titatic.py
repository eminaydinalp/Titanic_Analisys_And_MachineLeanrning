# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:18:57 2020

@author: USER
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
    
train_data = pd.read_csv("train.csv")
test_data  = pd.read_csv("test.csv") 

train_data.head()
train_data.columns
train_data.describe()
train_data.info()

# 1.Univariate Variable Analysis

# 1.1 Categorical Variable

def bar_plot(degisken):
    
    var = train_data[degisken]
    adet = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(adet.index, adet)
    plt.xticks(adet.index, adet.index.values)
    plt.ylabel("Frequency")
    plt.title(degisken)
    plt.show()
    print("{} : \n {} ".format(degisken,adet))
 


# # degisken = train_data["Sex"]
# # adet = degisken.value_counts()
# # plt.figure(figsize = (9,3))
# # plt.bar(adet.index, adet)

categori = ["Survived", "Pclass", "Sex", "Embarked", "SibSp", "Parch"]
for i in categori:
    bar_plot(i)
    


#bar_plot("Sex")

# 1.2 Numerical Variable
    
def plot_hist(degisken):
    plt.figure(figsize = (9,3))
    plt.hist(train_data[degisken])
    plt.xlabel(degisken)
    plt.ylabel("Frequency")
    plt.title("{} adfa ".format(degisken))
    plt.show()
    
    
numerik = ["Fare","Age","PassengerId"]
for i in numerik:
    plot_hist(i)
    

# Basic Data Analysis

# Pclass - Survived
# Sex - Survived
# SibSp - Survived
# Parch - Survived


train_data[["Pclass","Survived","Sex"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_data[["SibSp","Survived","Sex"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
a = train_data.groupby(["Pclass"]).mean()
b = train_data.groupby(["Sex"]).mean()
train_data[["Survived","SibSp"]].groupby(["SibSp"]).mean()
c = train_data[["Survived","Pclass","Sex"]].groupby(["Sex","Pclass"]).mean()


# Outlier Detection

def outliers(data, ozellik):
    
    outliers_index = []
    
    for i in ozellik:
        
        Q1 = np.percentile(data[i], 25)
        
        Q3 = np.percentile(data[i], 75)
        
        IQR = Q3 - Q1
        
        outlier_step = IQR * 1.5
        
        outlier_list_col = data[(data[i] < Q1 - outlier_step) | (data[i] > Q3 + outlier_step)].index
        
        outliers_index.extend(outlier_list_col)
        
    outliers_index = Counter(outliers_index)
    
    multiple_outliers = list(i for i, v in outliers_index.items() if v > 2)
    
    return multiple_outliers

outlier1 = train_data.loc[outliers(train_data, ["Age","SibSp","Parch","Fare"])]
outlier2 = train_data.loc[outliers(train_data, ["SibSp","Parch","Fare"])]
outlier3 = train_data.loc[outliers(train_data, ["Age","Parch","Fare"])]

# drop outliers
train_data = train_data.drop(outliers(train_data,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)   

# Missing Value

train_df_len = len(train_data)
train_data = pd.concat([train_data,test_data],axis = 0).reset_index(drop = True) 
train_data.head()

# Find Missing Value

train_data.columns[train_data.isnull().any()]
train_data.isnull().sum()

# Fill Missing Value

train_data[train_data["Embarked"].isnull()]  
train_data.boxplot(column="Fare",by = "Embarked")

train_data["Embarked"] = train_data["Embarked"].fillna("C")
train_data[train_data["Embarked"].isnull()]  

train_data[train_data["Fare"].isnull()]

x = train_data[train_data["Pclass"] == 3]["Fare"].mean()
train_data["Fare"] = train_data["Fare"].fillna(x)
train_data[train_data["Fare"].isnull()]

# Visualization

list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_data[list1].corr(), annot = True, fmt = ".2f")
plt.show()

g = sns.factorplot(x = "SibSp", y = "Survived", data = train_data, kind = "bar", size = 6)
#g.set_ylabels("Survived Probability")

sns.factorplot(x = "Pclass", y = "Survived", data = train_data, hue = "Sex", kind = "bar")
sns.factorplot(x = "Pclass", y = "Fare", data = train_data, hue = "Sex", kind = "violin")

sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_data, size = 6)

      
g = sns.FacetGrid(train_data, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)

g = sns.FacetGrid(train_data, col = "Survived", row = "Pclass")
g.map(plt.hist, "Age", bins = 25)

g = sns.FacetGrid(train_data, row = "Embarked")
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()

g = sns.FacetGrid(train_data, row = "Embarked")
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
    
g = sns.FacetGrid(train_data, row = "Embarked", col = "Survived", size = 2.3)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()


# Fill Missing: Age Feature
   
x = train_data[train_data["Age"].isnull()]
sns.factorplot(x = "Sex", y = "Age", data = train_data, kind = "box")
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass",data = train_data, kind = "box")

sns.factorplot(x = "Parch", y = "Age", data = train_data, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_data, kind = "box")

sns.heatmap(train_data[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)
    
index_nan_age = list(train_data["Age"][train_data["Age"].isnull()].index)
for i in index_nan_age:
    
    age_pred = train_data["Age"][((train_data["SibSp"] == train_data.iloc[5]["SibSp"]) &(train_data["Parch"] == train_data.iloc[5]["Parch"])& (train_data["Pclass"] == train_data.iloc[5]["Pclass"]))].median()
    
    age_med = train_data["Age"].median()
    
    if not np.isnan(age_pred):
        train_data["Age"].iloc[i] = age_pred
    else:
        train_data["Age"].iloc[i] = age_med


train_data[train_data["Age"].isnull()]


name = train_data["Name"].head(10)
train_data["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]

train_data = pd.get_dummies(train_data, columns=["Embarked"])

# Modeling
test = train_data[train_df_len:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)
test.head()

train = train_data[:train_df_len]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))


# Simple Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

    
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]    
    

cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")

# Ensemble Modeling¶

votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))




















