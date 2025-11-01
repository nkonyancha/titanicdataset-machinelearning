#%%
##random forest classifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

#%%

df=pd.read_csv('train.csv')

def clean_data(dataframe):
    dataframe = dataframe.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    for fare in dataframe['Pclass'].unique():
        median_fare=dataframe[dataframe['Pclass']==fare]['Fare'].median()
        dataframe.loc[(dataframe['Fare']==0) & (dataframe['Pclass']==fare), 'Fare']=median_fare
    for pclass in dataframe['Pclass'].unique():
        median_age=dataframe[dataframe['Pclass']==pclass]['Age'].median()
        dataframe.loc[(dataframe['Age'].isnull()) & (dataframe['Pclass']==pclass), 'Age']=median_age
    dataframe['Embarked'].fillna(dataframe['Embarked'].mode()[0], inplace=True)
    dataframe["Sex"]= dataframe["Sex"].map({'male':0,'female':1})
    dataframe=dataframe.drop(index=dataframe[dataframe['Embarked'].isnull()].index)
    dataframe["Embarked"]= dataframe["Embarked"].map({'S':0,'C':1,'Q':2})
    return dataframe


cleaned_data = clean_data(df)
X = cleaned_data.drop(columns=['Survived','PassengerId'], axis=1)
y = cleaned_data['Survived']
model = DecisionTreeClassifier(random_state=42)

def evaluate_model(model, X, y, k=5):
    """
    Evaluate a classification model using K-Fold cross-validation.
    Returns mean Accuracy, F1-score, and ROC-AUC.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    acc = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y, cv=kfold, scoring='f1').mean()
    roc = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc').mean()
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean Accuracy: {acc:.3f}")
    print(f"Mean F1-score: {f1:.3f}")
    print(f"Mean ROC-AUC: {roc:.3f}")
evaluate_model(model, X, y)
model.fit(X,y)

testdata=pd.read_csv('test.csv')
tdata=clean_data(testdata)

tdata['Age']= tdata['Age'].fillna(tdata.groupby('Pclass')['Age'].transform('median'))
tdata['Fare']=tdata['Fare'].fillna(tdata[tdata['Pclass']==3]['Fare'].median())

predictions=model.predict(tdata.drop(columns=['PassengerId']))
output=pd.DataFrame({'PassengerId':tdata['PassengerId'],'Survived':predictions})
output.to_csv('submission.csv',index=False)
print("Submission file created: submission.csv")
# %%
