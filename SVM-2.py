# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:22:25 2022

@author: Zahran
"""

# 1) importing the needed libraries
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization

# 2) loading and defining the dataset

data_train_pure = pd.read_csv('D:\\Selected 1\\Archive\\train.csv')
data_test_pure = pd.read_csv('D:\\Selected 1\\Archive\\test.csv')

# 3) The normalization of the dataset

# 3.1) removing the unnecessarily columns from the dataset
data_train = data_train_pure.drop('price_range', axis=1)
data_test = data_test_pure.drop('id', axis=1)

# 3.2) getting the Y values
Y = data_train_pure['price_range']
# 3.3) Start the normalization of the dataset
std = StandardScaler()
# 3.3.1) scaling of the train dataset
data_train_std = std.fit_transform(data_train)

# 3.3.2) scaling of the test dataset
data_test_std = std.transform(data_test)

# 4) Training the SVM model
model = SVC(C=100, kernel = 'linear' ,probability=True,break_ties=True , random_state=50)
# 4.1) Training phase
model.fit(data_train_std, Y)
# 4.2) predicting the result phase
print(model.predict(data_test_std))

# 5) Training By splitting the train dataset to get the accuracy

# 5.1) Splitting the Train dataset into test and train (0.2 to get 400 row ) random to give me a random rows
data_train_split, data_test_split, Y_train, Y_Test = train_test_split(data_train, Y, test_size=0.2)

# 5.2) Normalization of the new  dataset

# 5.2.1) scaling the new train dataset
data_train_std2 = std.fit_transform(data_train_split)

# 5.2.2) scaling the new test dataset
data_test_std2 = std.transform(data_test_split)

# 5.3) Trying again with the split dataset
model2 = SVC(C=100, kernel = 'linear' ,probability=True,break_ties=True, random_state=50)
# 5.3.1) Training phase
model2.fit(data_train_std2, Y_train)
# 5.3.2) predicting the phase
Y_prediction = model2.predict(data_test_std2)
print(Y_prediction)

# 6) The accuracy test
model_accuracy = accuracy_score(Y_Test, Y_prediction)
print('\nAccuracy : ',model_accuracy)


# 7) Confusion Matrix
cm = multilabel_confusion_matrix(Y_Test, Y_prediction)
print("Confusion Matrix :",multilabel_confusion_matrix(Y_Test, Y_prediction))
print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# 8) Sensitivity And Specificity
sensitivity = recall_score(Y_Test , Y_prediction , average = 'macro')
specificity = recall_score(np.logical_not(Y_Test) , np.logical_not(Y_prediction) , average = 'macro')

print("\nSensitivity : ",sensitivity)
print("\nSpecificity : ",specificity)

# 9) Precision Score

print("\nPrecision Score :",precision_score(Y_Test, Y_prediction, average='macro')) 

# 10) Recall Score
print("\nRecall Score :",recall_score(Y_Test, Y_prediction, average='macro')) 

# 11) ROC Curve
fpr, tpr, thresholds = roc_curve(Y_Test, Y_prediction ,pos_label=1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC Curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


#=============================================================================