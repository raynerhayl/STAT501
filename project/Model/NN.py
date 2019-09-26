from numpy import loadtxt
import pandas as pd
import collections
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib


# data = pd.read_csv('data/data_raw.csv', sep=',')
#
# cat_columns = ["DRG2", "GenderCode", "Ethnicity", "Domicile", "Residency", "arrival_dow","Posttake_dow", "PrincipalDiagnosis2",
#                "month", "season", "GenMedAdmission", "EthnicityGroupMPIO","PatientAgeGroup10YearCategory","Team","Breached6HourIndicator",
#                "RestHomeFlag", "DischargeDoctorID", "LengthOfStay", "ICUOccupiedHrs", "MET_CardiacArrest", "METNumber","MET_ImmediateOutcome", "KenepuruTransfer"]
# drop_columns = ["PatientID", "PrincipalDiagnosis", "elx_sum", "DomicileDesc", "DRG","PatientAgeGroup3SplitCategory", "PatientID"]
# data = data.drop(drop_columns, axis = 1)
# data = pd.get_dummies(data, prefix_sep="__", columns=cat_columns)
#
# data.to_csv("data/data_encoded.csv", index=False)

def get_train_test(data):
    X = data.drop(['ReadmittedIn14Days'], axis=1)
    y = data['ReadmittedIn14Days']

    names = list(X)
    print(names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)

    print("Before rebalance: ", y_train.value_counts())

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("X_train Shape : ", X_train.shape)
    print("X_test Shape : ", X_test.shape)

    X_train = pd.DataFrame.from_records(X_train, columns=names)
    X_test = pd.DataFrame.from_records(X_test, columns=names)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test

def test_Model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    #y_pred = (y_pred > 0.5)
    #cm = confusion_matrix(y_test, y_pred)

    #print(cm)
    #print(classification_report(y_test, y_pred))

    return(y_pred)

def get_NN(X_train, y_train):
    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add((Dropout(0.5)))
    model.add(Dense(100, activation='relu'))
    model.add((Dropout(0.5)))
    model.add(Dense(200, activation='relu'))
    model.add((Dropout(0.5)))
    model.add(Dense(200, activation='relu'))
    model.add((Dropout(0.5)))
    model.add(Dense(100, activation='relu'))
    model.add((Dropout(0.5)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=100)
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)

    return model

data = pd.read_csv('data/data_encoded.csv', sep=',')
X_train, X_test, y_train, y_test = get_train_test(data)
print(collections.Counter(y_test))
model = get_NN(X_train, y_train)
y_pred = test_Model(model, X_test, y_test)
print(roc_auc_score(y_test, y_pred) * 100)

# ##train model
# rnd_clf = RandomForestClassifier()
# rnd_clf.fit(X_train, y_train)
#
# ##cross validation of model result using roc_auc as performance metrics
#
# CV_score = cross_val_score(minmax_clf, X_train, y_train, cv=10, scoring='roc_auc')
# print(CV_score)
#
# # try model on test data
# probas = rnd_clf.predict_proba(X_test)[:, 1]
# AUC = metrics.roc_auc_score(y_test, probas)
# print(AUC)
