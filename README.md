# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SHIV SUJAN S R
RegisterNumber:  212223040194
*/
```
```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset, specifying the encoding as 'latin-1'
# 'latin-1' is a common encoding that often resolves this error.
# If this doesn't work, try other encodings like 'ISO-8859-1', 'cp1252', etc.
data = pd.read_csv('spam.csv', encoding='latin-1')

# Print the column names to verify the correct column name
print(data.columns)

# Assuming the column name for the text is 'v2' based on the DataFrame information
# Adjust this accordingly if it's different
X = data['v2']
y = data['v1']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```
## Output:

![376629762-20e938cc-66f7-40fa-ae3c-0356da95c16e](https://github.com/user-attachments/assets/fbd8b075-3dc7-4d13-abd0-a00c5b301872)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
