#Step 1: Build Dataset

# Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set datas              1        2         3      4      5      6       7        8        9         10
data = {
    'customer_name' : [ 'Jay', 'Valerie', 'Dits', 'Vel', 'Ara', 'Jan', 'Sofia', 'Kylie', 'Jayson', 'Lovely',
#                        11       12         13         14       15        16        17        18         19       20
                      'Charish', 'Karla', 'Rachel', 'Jamilla', 'Jerah', 'Effamar', 'Juan', 'Francis', 'Charles', 'Ryzel'],

# browse_time            1   2   3   4   5   6   7  8   9   10  11  12  13  14  15  16  17  18  19  20 = 20 raw data
    'browse_time_min' : [10, 30, 50, 60, 40, 20, 5, 20, 40, 60, 30, 10, 50, 40, 50, 10, 15, 25, 35, 55],

# attendance rate       1    2    3    4    5    6    7    8    9    10   11   12   13   14   15    16   17   18   19    20 = 20 raw data
    'cart_value_php' : [110, 130, 150, 170, 190, 200, 220, 240, 260, 280, 300, 500, 550, 1100, 600, 700, 450, 360, 1000, 950],

# buyers          1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 = 20 raw data
    'purchased': [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
df['status'] = df['purchased'].map({1: 'buyer', 0: 'non-buyer'})

print(df[['customer_name', 'browse_time_min', 'cart_value_php', "status"]])

#Step 2: Split

# Features
X = df[['browse_time_min', 'cart_value_php']]

# Labels
y = df['purchased']
names = df['customer_name']

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, names, test_size=0.25, random_state=42)

# Fetch Training Group
print("Training Group (model study this record)")
print(list(names_train))

# Fetch Testing Group
print("\n Testing Group (model has never seen their outcome): ")
print(list(names_test))

#Step 3: TRain the model

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Model trained on", len(X_train), "customers!")

#Step 4: Predict

from pandas.core import series
y_pred = model.predict(X_test)

results = pd.DataFrame({
    'customer_name': names_test.values,
    'actual' : y_test.map({1: 'buyer', 0: 'At Risk'}).values,
    'predicted' : pd.Series(y_pred).map({1: 'buyer', 0: 'non-buye'}).values
})

print(results)

#Step 5: Accuracy

acc = accuracy_score(y_test, y_pred)

print("Accuracy", round(acc*100, 2), "%")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['non-buyer', 'buyer']))

# SUPPORTING QUESTION AND ANSWER

# **ACTIVITY QUESTIONS**

# 1. How many customers are "Non-Buyer" vs "Buyer"?
#  - Based on **Step 1: Dataset** there are **9 Non-Buyers** and **11 Buyers**

# 2. What's the average browsing time for "Non-Buyer" customers? For "Buyer" customers?

# 3. What's the average cart value for each group?

# 4. Are there any suspicious patterns? (Customers with similar behavior but different outcomes?)
#  - the 1.00 accuracy

# 5. How many customers are in the training set? Testing set?
#  - 15 in training and 5 in testing

# 6. Which customers are in the test set (the "exam" the model hasn't seen)?
#  - Jay, Francis, Effamar, Valerie, Jayson

# 7. What's the model's accuracy?
#  - 100.0%

# 8. Did it make any mistakes? Which customers?
#  - I think it doesn't

# 9. Look at the confusion matrix – how many "Non-Buyer" customers were correctly identified?
#  - 1 non buyer

# 10. Before running the code, predict: which customers will be "Non-Buyer"?
#  - I think 10

# 11. Should the model be more conservative (only predict "Buyer" when very confident) or aggressive (predict "Buyer" more often)? Why?
#  - No, it should both, so it can focus how many buyers will receive a discount

# 12. If Shopee sends discount vouchers only to predicted "Buyers," what's the risk of the model making a mistake? (Think about false positives and false negatives)
#  - It will give discount for non-buyer customer even they don't deserves it, also it will be unfair for buyers

# 13. What if the model systematically predicts lower-income customers (smaller cart values) as "Non-Buyer" and never sends them discounts, trapping them in a cycle of higher prices? Is this fair?
#  - Yes it will be fair because buyers helps the seller and they deserve a discounts, unlike of non-buyers
