# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

titanic = pd.read_csv('../input/train.csv')

titanic.isnull().sum()
titanic.describe()

#convert male to 1 female to 0 
titanic['Sex'].replace({'female':0,'male':1}, inplace=True)


sns.countplot(x='Survived', data=titanic)
plt.show()

columns=titanic.columns[:10]
plt.subplots(figsize=(20,20))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    titanic[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()


titanic1=titanic[titanic['Survived']==1]
columns=titanic.columns[:10]
plt.subplots(figsize=(20,20))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    titanic1[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()

sns.pairplot(data=titanic, hue='Survived',diag_kind='kde')
plt.show()

survived = titanic['Survived']
#data = titanic[titanic.columns[2:10]]
data = titanic[['Pclass','Sex','Age','SibSp', 'Parch', 'Ticket', 'Fare']]

#stratify ensures equal proportions of surived in test and training set
train,test = train_test_split(titanic, test_size=0.25, random_state=0, stratify = titanic['Survived'])

train_X  =train[['Pclass','Sex','SibSp', 'Parch', 'Fare']]
test_X = test[['Pclass','Sex','SibSp', 'Parch', 'Fare']]
train_Y = train['Survived']
test_Y = test['Survived']



types = ['rbf', 'linear']
for i in types:
    model = svm.SVC(kernel = i)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    print('Accuracy for SVM kernel = ',i,'is', metrics.accuracy_score(prediction,test_Y))

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))


model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_Y))


a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
plt.show()
print('Accuracies for different values of n are:',a.values)



#without standardisation
abc=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    abc.append(metrics.accuracy_score(prediction,test_Y))
models_dataframe=pd.DataFrame(abc,index=classifiers)   
models_dataframe.columns=['Accuracy']
models_dataframe

sns.heatmap(titanic[titanic.columns[2:10]].corr(),annot=True,cmap='RdYlGn')

from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100,random_state=0)
X=titanic[['Pclass','Sex','SibSp', 'Parch', 'Fare']]
Y=titanic['Survived']
model.fit(X,Y)
pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)

# Use only important features indicated by random forest classifier
# transform features to a standard normal dist
from sklearn.preprocessing import StandardScaler #Standardisation

# important features + survival outcome
titanic2 = titanic[['Fare','Sex','Pclass','Survived']]

# isolate features
features = titanic2[titanic2.columns[:3]]

# Gaussian Standardisation of important features
features_standard = StandardScaler().fit_transform(features)

#put these features into a dataframe
x = pd.DataFrame(features_standard,columns=[['Fare','Sex','Pclass']])

#re-split data into train and test sets
x['Survived'] = titanic2['Survived']
survived = x['Survived']
train1,test1 = train_test_split(x,test_size=0.25,random_state=0,stratify=x['Survived'])
train_X1=train1[train1.columns[:3]]
test_X1=test1[test1.columns[:3]]
train_Y1=train1['Survived']
test_Y1=test1['Survived']

abc = []
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(train_X1,train_Y1)
    prediction=model.predict(test_X1)
    abc.append(metrics.accuracy_score(prediction,test_Y1))
new_models_dataframe=pd.DataFrame(abc,index=classifiers)   
new_models_dataframe.columns=['New Accuracy']  

new_models_dataframe = new_models_dataframe.merge(models_dataframe,left_index=True,right_index=True,how='left')
new_models_dataframe['Change']=new_models_dataframe['New Accuracy']-new_models_dataframe['Accuracy']
new_models_dataframe