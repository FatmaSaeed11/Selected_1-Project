# 1) importing the needed libraries

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy as sp

# 2) loading and defining the dataset

data_train_pure = pd.read_csv('kaggle/input/mobile-price-classification/train.csv')
data_test_pure = pd.read_csv('kaggle/input/mobile-price-classification/test.csv')

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

# 4) Training the Logistic regression model
logistic_model = LogisticRegression()
# 4.1) Training phase
logistic_model.fit(data_train_std, Y)
# 4.2) predicting the result phase
print(logistic_model.predict(data_test_std))

# 5) Training By splitting the train dataset to get the accuracy

# 5.1) Splitting the Train dataset into test and train (0.2 to get 400 row ) random to give me a random rows
data_train_split, data_test_split, Y_train, Y_Test = train_test_split(data_train, Y, test_size=0.2, random_state=1)

# 5.2) Normalization of the new  dataset

# 5.2.1) scaling the new train dataset
data_train_std2 = std.fit_transform(data_train_split)

# 5.2.2) scaling the new test dataset
data_test_std2 = std.transform(data_test_split)

# 5.3) Trying again with the split dataset
logistic_model2 = LogisticRegression()
# 5.3.1) Training phase
logistic_model2.fit(data_train_std2, Y_train)
# 5.3.2) predicting the phase
Y_prediction = logistic_model2.predict(data_test_std2)
print(Y_prediction)

# 6) The accuracy test
logistic_model_accuracy = accuracy_score(Y_Test, Y_prediction)
print('Accuracy  : ',logistic_model_accuracy)

# 7) confusion matrix
confuison_matrix=metrics.confusion_matrix(Y_Test , Y_prediction)
print("Confusion matrix")
print(confuison_matrix)
# 8)sensitivity and specificity
sensitivity1 = confuison_matrix[0,0]/(confuison_matrix[0,0]+confuison_matrix[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = confuison_matrix[1,1]/(confuison_matrix[1,0]+confuison_matrix[1,1])
print('Specificity : ', specificity1)

