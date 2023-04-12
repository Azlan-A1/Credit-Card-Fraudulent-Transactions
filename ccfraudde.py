#Credit Card Fraud Detection

#pandas is used for data manipulation and analysis 
#numpy is used for adding support for larger multi-dimensional arrays as well as matrices. Essentially dealing 
#with more data 

#importing various python libraries for array creation and manipulation 

import numpy as np #NumPy
import pandas as pd #Pandas
from sklearn.model_selection import train_test_split #scikit-learn
from sklearn.linear_model import LogisticRegression #scikit-learn
from sklearn.metrics import accuracy_score #scikit-learn
from sklearn.metrics import confusion_matrix #scikit-learn


#A '0' represents normal transaction
#A '1' represents fraud/fake transaction

#loads the data received from the file onto the data frame
#data frame can be implemented due to pandas python library 

#system reads the file 
creditcard_data = pd.read_csv('credit_data.csv')

#displays first 5 rows of Dataframe
creditcard_data.head()

#displays the last few rows of Dataframe
creditcard_data.tail()

#displays information about the Dataframe
creditcard_data.info()

#checks missing values if there are any present and counts them
creditcard_data.isnull().sum()


#demonstrates the amount of real transactions and fraudulent transactions
creditcard_data['Class'].value_counts()

#organizes the data to analyze 
existent = creditcard_data[creditcard_data.Class == 0]
nonexistent = creditcard_data[creditcard_data.Class == 1]

#displays authorized transactions vs illegitimate transactions
print(existent.shape)
print(nonexistent.shape)


#used to describe numerical statistics such as mean, min, max 
existent.Amount.describe()



nonexistent.Amount.describe()


#used to group rows of Dataframe, indicates whether transaction info states fraudulent or non-fraudulent
#helps identify the patterns of fraudulent transactions that are separate from authorized ones
creditcard_data.groupby('Class').mean()


#number of fraudulent transactions is 492. the value n represents that.
sample_existent = existent.sample(n=492)


#joins Dataframes together, allows you to join two dataframes together and print them using one variable in the end
advanced_infoset = pd.concat([sample_existent, nonexistent], axis=0)


#first few rows of Dataframe
advanced_infoset.head()


#last few rows of Dataframe
advanced_infoset.tail()

#same as line 40, but with joined dataframes
advanced_infoset['Class'].value_counts()

#joined dataframes
advanced_infoset.groupby('Class').mean()


#demonstrates rows and columns displaying number, time, purchased amount etc. 
X = advanced_infoset.drop(columns='Class', axis=1)

#shows dataset with normal transactions and fraudulent ones
Y = advanced_infoset['Class']


print(X)


print(Y)

#splits the training and testing data into their own categories

X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size=0.32, stratify=Y, random_state=3)

print(X.shape, X_training.shape, X_testing.shape)


print(X.shape, X_training.shape, X_testing.shape)

model = LogisticRegression()

#Logistic Regression Model with Training Data
model.fit(X_training, Y_training)



#displaying accuracy for the training data
Xtrain_prediction = model.predict(X_training)
trainingd_accuracy = accuracy_score(Xtrain_prediction, Y_training)

print('The Accuracy on the Training data is : ', trainingd_accuracy)


#displaying accuracy for the testing data
Xtest_prediction = model.predict(X_testing)
testd_accuracy = accuracy_score(Xtest_prediction, Y_testing)

#accuracy score 
print('The Accuracy score on the Test Data is : ', testd_accuracy)