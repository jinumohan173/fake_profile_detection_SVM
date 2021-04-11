# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:21:33 2021

@author: Jinu
"""


# coding: utf-8

### Detect fake profiles in online social networks using Support Vector Machine

# In[57]:
    

# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import detector as gender
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

#from sklearn import cross_validation

from sklearn.model_selection import cross_validate


from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
#from sklearn.cross_validation import StratifiedKFold, train_test_split

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split ,StratifiedKFold

#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
#from sklearn.learning_curve import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
get_ipython().magic(u'matplotlib inline')


pkl_filename = "pickle_model.pkl"
####### function for reading dataset from csv files

# In[58]:

#D:/excellence/fake_profile_identification/Fake-Profile-Detection-using-ML-master/data/fusers.csv    
#D:\excellence\fake_profile_identification\Fake-Profile-Detection-using-ML-master\data\fusers.csv

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("data/nam_dict.csv")
   # fake_users = pd.read_csv("data/fusers.csv")
    # print genuine_users.columns
    # print genuine_users.describe()
    #print fake_users.describe()
  #  x=pd.concat([genuine_users,fake_users])   
   # y=len(fake_users)*[0] + len(genuine_users)*[1]
    print("xvalue ",genuine_users)
    #print("yvalue ",y)
    return genuine_users
    


####### function for predicting sex using name of person

# In[59]:

def predict_sex(name):
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code


####### function for feature engineering

# In[62]:

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
    x=x.loc[:,feature_columns_to_use]
    print("xxxxx",x)
    return x


####### function for ploting learning curve

# In[63]:


####### function for plotting confusion matrix

# In[65]:




####### function for plotting ROC curve

# In[71]:




####### Function for training data using Support Vector Machine

# In[72]:

def split_data(X_train,y_train,X_test):
    """ Trains and predicts dataset with a SVM classifier """
    
    print("xtest before ",X_test)
    # Scaling features
    X_train=preprocessing.scale(X_train)
    X_test=preprocessing.scale(X_test)
    
    print("xtest " ,X_test)

    Cs = 10.0 ** np.arange(-2,3,.5)
    gammas = 10.0 ** np.arange(-2,3,.5)
    param = [{'gamma': gammas, 'C': Cs}]
    cvk = StratifiedKFold(n_splits=10)
    #cvk = StratifiedKFold(y_train,5)
    classifier = SVC()
    clf = GridSearchCV(classifier,param_grid=param,cv=cvk)
    clf.fit(X_train,y_train)
    print("The best classifier is: ",clf.best_estimator_)
    clf.best_estimator_.fit(X_train,y_train)
    classifier = clf.best_estimator_.fit(X_train,y_train)
 
    # Predict class
    y_pred = clf.best_estimator_.predict(X_test)
    with open(pkl_filename, 'wb') as file:
        
        pickle.dump(classifier, file)
    return y_test,y_pred


# In[76]:

print ("reading datasets.....\n")
x=read_datasets()


# In[77]:

print ("extracting featues.....\n")
x=extract_features(x)
print (x.columns)
print (x.describe())


# In[78]:
    


print ("spliting datasets in train and test dataset...\n")
#X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)
X_train = train_test_split(x,shuffle=False)#, y, test_size=0.20, random_state=44)

X_train,X_test = train_test_split(x, test_size=0.20, random_state=44)
X_train=preprocessing.scale(X_train)
print("xtrain ",X_train[0])
print("type ..", type(X_train))

print()


with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


"""
# In[79]:

print ("training datasets.......\n")
y_test,y_pred = split_data(X_train,y_train,X_test)


# In[80]:




    
# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
"""
print("xtrin[0] ",X_train[0])
Ypredict = pickle_model.predict(X_train)
print("y predict ..",Ypredict)

