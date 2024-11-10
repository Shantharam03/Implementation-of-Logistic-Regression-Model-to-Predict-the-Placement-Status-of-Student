# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: shantharam . M
RegisterNumber:  24900113
*/
```
import pandas as pd

data=pd.read_csv("Placement_Data.csv")

print("HEAD")

print(data.head())

data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis=1)

print("COPY")

print(data1.head())

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])

data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"])

data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"])

data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])

data1["status"]=le.fit_transform(data1["status"])

print("FIT TRANSFORM")

print(data1)

x=data1.iloc[:,:-1]

x

y=data1["status"]

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("LOGISTIC REGRESSION")

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver="liblinear")

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)


print(y_pred)

print("ACCURACY SCORE")

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

print(accuracy)

print("CONFUSION MATRIX")

from sklearn.metrics import confusion_matrix

confusion=confusion_matrix(y_test,y_pred)

print(confusion)

print("CLASSIFIFCARION REPORT")

from sklearn.metrics import classification_report

classification_report1=classification_report(y_test,y_pred)

print(classification_report1)

print("PREDICTION")

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
## Output:
![Screenshot 2024-11-10 191326](https://github.com/user-attachments/assets/6fbf6fcc-ee7e-4ebd-98ce-de0cdb003b15)

![Screenshot 2024-11-10 191402](https://github.com/user-attachments/assets/4883983c-a3b1-46d6-a9cc-82e963549192)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
