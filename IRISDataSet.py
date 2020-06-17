# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:47:09 2020
Hello World for Machine Learning Simple Project (looking at IRIS dataset)

@author: 17jlo
"""



from pandas import read_csv
from pandas.plotting import scatter_matrix 
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading Datasets
#UCI DataSets very good comma separated 

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# can also download into the working directory and call local file name
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # column names
dataset = read_csv(url, names = names) 

# useful commands to look at the data 

# shape 
print(dataset.shape) # (instances,attributes)

# head 
print(dataset.head(20)) # prints the top of the data first 20 rows 

#descriptions
print(dataset.describe())

# class distribution 
'Returns the number of rows that appear in each class'
print(dataset.groupby('class').size())

'Univariate plots for singular variables'

# box and whisker plots 
dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey = False)
pyplot.show()

#histograms 
'certain algorithmns can exploit gaussian assuumptions'
dataset.hist()
pyplot.show()

# scatter plot matrix 

scatter_matrix(dataset)
pyplot.show()

'creating models and looking at algorithmns'

# Split-out validation dataset

array = dataset.values
X = array[:, 0:4] # input
Y = array[:,4]  # output 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 1)

'there are some 5.2 test harness background sources to read'

# Checking the Algorithmns

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))

# evaluates each model in turn 
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model,X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

'closest to 1 is the best model it seems '

#comparing algorithms

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


'valiation data sets are useful and helps when training in case issues'

# predictions on the validation dataset

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
'can also make predictions on single rows of data '
'can also save for later '

#evaluating predictions 
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

'f1-score and support used at evaluation criteria and are very good here'
'accuracy is at the topp 0.9666667'

