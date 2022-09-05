import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ds=pd.read_csv("C:\\Users\\hp\\Documents\\GitHub\\py\ML\\9_decision_tree\\Exercise\\titanic.csv")
Y=ds['Survived']
nd=ds.drop(['Fare','Name','SibSp','Ticket','Cabin','Embarked','Parch','PassengerId'],axis='columns')
from sklearn.preprocessing import LabelEncoder
ns=LabelEncoder()
nd['NS']=ns.fit_transform(nd['Sex'])
DTS=nd.drop(['Sex','Survived','Age'],axis='columns')
model=DecisionTreeClassifier()
a,b,c,d=train_test_split(DTS,Y,test_size=0.3)
model.fit(a,c)
rv=model.score(b,d)
print(rv)
dfn=nd.groupby('Survived').mean()
