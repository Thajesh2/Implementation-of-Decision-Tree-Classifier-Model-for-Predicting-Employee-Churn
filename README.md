# Ex No:8 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## Thajesh  K
## 212223230229
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
   
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Thajesh  K
RegisterNumber: 212223230229
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("REG-NO: 212224040088")
print("NAME: EZHILARASI N")
print("Accuracy:",accuracy)
print("Confusion:",confusion)
print("Classification:",classification)

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:

### Data Head:
<img width="1372" height="222" alt="image" src="https://github.com/user-attachments/assets/77d69064-c9d3-41f7-99a8-57dd6cd4b52d" />


### Dataset info :
<img width="712" height="407" alt="image" src="https://github.com/user-attachments/assets/f68460fd-e27d-405f-be7f-e826a5ba6ea1" />


### Null Dataset:
<img width="368" height="267" alt="image" src="https://github.com/user-attachments/assets/8f0383ec-aec2-4458-8796-adf9e53a2f4e" />


### Values count in left column:
<img width="360" height="88" alt="image" src="https://github.com/user-attachments/assets/273736ff-42e7-4715-8e08-0634c04b0273" />


### Dataset transformed head:
<img width="1367" height="213" alt="image" src="https://github.com/user-attachments/assets/fa020efe-9a81-40a7-a030-da4f86aaff1b" />


### x.head:
<img width="1247" height="216" alt="image" src="https://github.com/user-attachments/assets/231ea7b4-a706-435b-817b-fda213dc1742" />


### Data prediction:
<img width="1372" height="126" alt="image" src="https://github.com/user-attachments/assets/25c5c614-285b-450e-a9f6-ea44d2840218" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
