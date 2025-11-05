import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load train data
df = pd.read_csv('train.csv')

# --- Data cleaning ---
def clean_data(dataframe):
    dataframe = dataframe.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    for fare in dataframe['Pclass'].unique():
        median_fare = dataframe[dataframe['Pclass'] == fare]['Fare'].median()
        dataframe.loc[(dataframe['Fare'] == 0) & (dataframe['Pclass'] == fare), 'Fare'] = median_fare
    for pclass in dataframe['Pclass'].unique():
        median_age = dataframe[dataframe['Pclass'] == pclass]['Age'].median()
        dataframe.loc[(dataframe['Age'].isnull()) & (dataframe['Pclass'] == pclass), 'Age'] = median_age
    dataframe['Embarked'].fillna(dataframe['Embarked'].mode()[0], inplace=True)
    dataframe["Sex"] = dataframe["Sex"].map({'male': 0, 'female': 1})
    dataframe = dataframe.drop(index=dataframe[dataframe['Embarked'].isnull()].index)
    dataframe["Embarked"] = dataframe["Embarked"].map({'S': 0, 'C': 1, 'Q': 2})
    return dataframe

cleaned_data = clean_data(df)
X = cleaned_data.drop(columns=['Survived', 'PassengerId'], axis=1)
y = cleaned_data['Survived']

# --- Evaluation function ---
def evaluate_model(model, X, y, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y, cv=kfold, scoring='f1').mean()
    prec = cross_val_score(model, X, y, cv=kfold, scoring='precision').mean()
    rec = cross_val_score(model, X, y, cv=kfold, scoring='recall').mean()

    print(f"Model: {model.__class__.__name__} {model.estimator.__class__.__name__}")
    print(f"Mean Accuracy: {acc:.3f}")
    print(f"Mean F1-score: {f1:.3f}")
    print(f"Mean Precision: {prec:.3f}")
    print(f"Mean Recall: {rec:.3f}")

# --- Bagging models ---
decisionbag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    random_state=42
)

logisiticbag = BaggingClassifier(
    estimator=LogisticRegression(max_iter=1000),
    n_estimators=50,
    random_state=42
)

# --- Evaluate both ---
evaluate_model(decisionbag, X, y)
evaluate_model(logisiticbag, X, y)

# --- Train both on full data ---
decisionbag.fit(X, y)
logisiticbag.fit(X, y)

# --- Load and clean test data ---
testdata = pd.read_csv('test.csv')
tdata = clean_data(testdata)
tdata['Age'] = tdata['Age'].fillna(tdata.groupby('Pclass')['Age'].transform('median'))
tdata['Fare'] = tdata['Fare'].fillna(tdata[tdata['Pclass'] == 3]['Fare'].median())

# --- Predict using bagged decision tree (you can switch to bag_lr if you want) ---
predictions = decisionbag.predict(tdata.drop(columns=['PassengerId']))

# --- Submission file ---
output = pd.DataFrame({'PassengerId': tdata['PassengerId'], 'Survived': predictions})
output.to_csv('submission.csv', index=False)
#print("✅ Submission file created: submission.csv")
