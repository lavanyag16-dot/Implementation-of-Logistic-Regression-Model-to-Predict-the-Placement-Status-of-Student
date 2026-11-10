# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.

2.Define the features (X) and target variable (y).

3.Split the data into training and testing sets.

4.Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: lavanya G

RegisterNumber: 21225230148 
*/
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```
## Output:
Head

<img width="750" height="190" alt="image" src="https://github.com/user-attachments/assets/eccffe74-6f1d-4f51-831e-85a347366803" />

Copy

<img width="750" height="223" alt="image" src="https://github.com/user-attachments/assets/f16692ec-6c1f-45fa-a000-6bd4368c53ff" />

FIT TRANSFORM

<img width="750" height="470" alt="image" src="https://github.com/user-attachments/assets/ee7cebce-968e-4d1e-bcb0-b9f707453e5a" />

LOGISTIC REGRESSION

<img width="750" height="186" alt="image" src="https://github.com/user-attachments/assets/6d56d081-7519-4908-94dc-21cd86cbe390" />

ACCURACY SCORE

<img width="750" height="99" alt="image" src="https://github.com/user-attachments/assets/7cc20665-c57a-429d-93f2-b64298afefe9" />

CONFUSION MATRIX

<img width="750" height="122" alt="image" src="https://github.com/user-attachments/assets/1b398c67-7fa5-4a16-a357-913404485ba2" />

CLASSIFICATION REPORT & PREDICTION

<img width="754" height="321" alt="image" src="https://github.com/user-attachments/assets/d0cbfe2d-1859-4cbb-b6a2-bd86f6161224" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
