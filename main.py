# A step by step example on using SVM with sklearn
# Authors: Wahab Hamou-Lhadj, Fatima AitMahammed, Mohammed Shehab
# SRT Lab

import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def start():
    # 1. Data loading
    dataset = pd.read_csv("data/weather.csv")

    # 2. Data preparation and preprocessing

    # 2.1 Converting categorical data into numerical data.
    # Note that we are treating the categorical features as nominal features.
    # You can also convert all attributes using apply(le.fit_transform).

    le = LabelEncoder()
    dataset['outlook'] = le.fit_transform(dataset['outlook'])
    dataset['temperature'] = le.fit_transform(dataset['temperature'])
    dataset['humidity'] = le.fit_transform(dataset['humidity'])
    dataset['windy'] = le.fit_transform(dataset['windy'])
    dataset['play'] = le.fit_transform(dataset['play'])

    # 2.2 Creating features and target labels
    features = dataset.drop(['play'], axis=1)
    targets = dataset['play']

    # 3. Spliting data
    # Here the data is plit into 70% training and 30% testing. By varying the test_size, you can change the training/testing ratio
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.3)

    # 4. Building a training model using svm (other classifiers can be used too)
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, y_train)

    # 5. Testing the model
    y_pred = clf.predict(X_test)

    # 6. Outputting test results
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)

    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))
    print("F1_Score: ", metrics.f1_score(y_test, y_pred))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# Start project
if __name__ == '__main__':
    start()
