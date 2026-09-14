#Step 1: build dataset

# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set datas              1        2         3      4      5      6       7        8        9         10
data = {
    'student_name' : [ 'Jay', 'Valerie', 'Dits', 'Vel', 'Ara', 'Jan', 'Sofia', 'Kylie', 'Jayson', 'Lovely',
#                        11       12         13         14       15        16        17        18         19       20
                      'Charish', 'Karla', 'Rachel', 'Jamilla', 'Jerah', 'Effamar', 'Juan', 'Francis', 'Charles', 'Ryzel'],

# study hours        1  2  3  4  5   6   7  8  9  10 11 12 13 14 15 16  17 18 19 20 = 20 raw data
    'study_hours' : [2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 4, 8, 2],

# attendance rate        1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20 = 20 raw data
    'attendance_rate' : [60, 62, 64, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 71],

# passed       1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 = 20 raw data
    'passed': [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
df['status'] = df['passed'].map({1: 'Passed', 0: 'At risk'})

print(df[['student_name', 'study_hours', 'attendance_rate', "status"]])

#Step 2: Split Data into Training and Testing Group

# Features
X = df[['study_hours', 'attendance_rate']]

# Labels
y = df['passed']
names = df['student_name']

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, names, test_size=0.25, random_state=42)

# Fetch Training Group
print("Training Group (model study this record)")
print(list(names_train))

# Fetch Testing Group
print("\n Testing Group (model has never seen their outcome): ")
print(list(names_test))

#Step 3: Train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Model trained on", len(X_train), "students!")

#Step 4: Predict

from pandas.core import series
y_pred = model.predict(X_test)

results = pd.DataFrame({
    'student_name': names_test.values,
    'actual' : y_test.map({1: 'Passed', 0: 'At Risk'}).values,
    'predicted' : pd.Series(y_pred).map({1: 'Passed', 0: 'At Risk'}).values
})

print(results)

#Step 5: Show model's accuracy

acc = accuracy_score(y_test, y_pred)

print("Accuracy", round(acc*100, 2), "%")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['At Risk', 'Passed']))


#SUPPORTING QUESTIONS AND ANSWERS
