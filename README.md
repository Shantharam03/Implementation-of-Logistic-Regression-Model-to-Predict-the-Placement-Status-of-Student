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
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
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
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
## HEAD
![Screenshot 2024-11-24 144206](https://github.com/user-attachments/assets/f729fe26-5aa2-4866-a949-45f664ff73c4)
## COPY
![Screenshot 2024-11-24 144316](https://github.com/user-attachments/assets/459b6e07-ff74-4b66-b4a2-770005c8d583)
## FIT TRANSFORM
![Screenshot 2024-11-24 144410](https://github.com/user-attachments/assets/13628126-f9cd-44e0-9b84-6986622fbd2c)
## LOGISTIC REGRESSION
![Screenshot 2024-11-24 144519](https://github.com/user-attachments/assets/86b7cd1e-82a5-4e7c-8d98-a0de1cd28bc8)
## ACCURACY SCORE
![Screenshot 2024-11-24 144606](https://github.com/user-attachments/assets/55748601-ed47-4acc-897b-b6d819a38e9a)
## CONFUSION MATRIX
![Screenshot 2024-11-24 144651](https://github.com/user-attachments/assets/eff6224f-350a-41c4-929e-170b339b9184)
## CLASSIFICATION REPORT
![Screenshot 2024-11-24 144815](https://github.com/user-attachments/assets/85b21aa8-1ae4-4961-98be-f9b97f57871a)
## PREDICTION
![Screenshot 2024-11-24 144852](https://github.com/user-attachments/assets/a8cb52fb-f9f9-4af6-aaaa-b31df611d36f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
