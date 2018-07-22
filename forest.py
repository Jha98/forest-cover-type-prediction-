import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn import cross_validation
import sklearn.linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

train = pd.read_csv("C:\\Users\\DEEPAK\\Downloads\\train.csv")
test = pd.read_csv("C:\\Users\\DEEPAK\\Downloads\\test.csv")
sub = pd.read_csv("C:\\Users\\DEEPAK\\Downloads\\sampleSubmission.csv")

sub = sub.iloc[:test.shape[0]]['Cover_Type']
test = pd.concat([test, sub], axis=1)

print("Shape of train",train.shape)
print("Shape of test", test.shape)
print("Shape of test+train", data.shape)
print(test.head())#to check cover_type is included in test 
print(data.describe())

print(data.dtypes)#No categorical data. All are numerical

print(data.columns.values)#56
data= data.iloc[:,1:]#to remove 1 column ie id
print(data.columns.values)



#print(train.info())

# How many samples of each cover type are there?

print(data["Cover_Type"].value_counts())#only one column
data["Cover_Type"].value_counts().plot(kind='bar',color='gold')
plt.ylabel("Number of Occurences")
plt.xlabel("Cover Type")
plt.show()

# How many of each Soil_Type are there?
A = np.array(data.columns.values)#40 columns
soil_types = [item for item in A if "Soil" in item]
for soil_type in soil_types:
	print (soil_type, train[soil_type].sum())

#we see that 'Soil_Type7', 'Soil_Type15' sum is 0 so we drop them
data=data.drop(['Soil_Type7','Soil_Type15','Soil_Type8','Soil_Type25'],axis=1)
print("columns:",data.columns.values)



y=data["Cover_Type"]
data=data.drop(["Cover_Type"],axis=1)
x=data


#LOGISTIC REGRESSION
reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
reg.fit(x_train,y_train)
print("LOGISTIC REGRESSION SCORE:",reg.score(x_test,y_test))#0.97

'''#ERROR
kfold=sklearn.model_selection.KFold(n_splits=10,random_state=7)
modelCV=sklearn.linear_model.LogisticRegression()
results=sklearn.model_selection.cross_val_score(modelCV,x,y,cv=kfold,scoring='accuracy')
results.mean()#0.62
print(results.mean())
'''


#RANDOM FOREST
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clf=sklearn.ensemble.RandomForestClassifier(random_state=0)
clf.fit(x_train,y_train)
print("RANDOM FOREST SCORE:",clf.score(x_test,y_test))#0.97



#LINEAR REGRESSION
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)#random_state=1)
reg=sklearn.linear_model.LinearRegression()
reg.fit(x_train,y_train)
print("LINEAR REGRESSION SCORE:",reg.score(x_test,y_test))#0.044


#a=15120+565892
#print("train",15120/a,"test=",565892/a)#train=0.02,test=0.97
'''
#SVM-TAKING A LOT OF TIME
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
model=sklearn.svm.SVC()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))#
'''

#POLYNOMIAL-degree2 or 3:score0.45

#DECISION TREE
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)

clf_gini=sklearn.tree.DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=3,min_samples_leaf=5)
clf=clf_gini.fit(x_train,y_train)
print("DECISION TREE SCORE:",clf_gini.score(x_test,y_test))#0.977
#print(clf_gini.n_features_)#50
sklearn.tree.export_graphviz(clf,out_file="tree5.dot")



#
knnc=KNeighborsClassifier(n_neighbors=5)
knnc.fit(x_train,y_train)
#y_knn_model=knnc.predict(x_test)
#score=accuracy_score(y_test,y_knn_model)
#print("SCORE OF KNeighborsClassifier= ",score)
print(knnc.score(x_test,y_test))


#svm
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
svmRBF=svm.SVC(kernel='rbf',gamma=1,C=1)
svmRBF.fit(x_train,y_train)
#y_svmSBF_model=svmRBF.predict(x_test)
#print(model.score(x_test,y_svmSBF_model))
print(svmRBF.score(x_test,y_test))
