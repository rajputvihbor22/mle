import pandas as pd
import matplotlib as plt
ds=pd.read_csv("C:\\Users\\hp\\Documents\\GitHub\\py\\ML\\9_decision_tree\\salaries.csv")
print(ds)
from sklearn.preprocessing import LabelEncoder
ncompany=LabelEncoder()
njoB=LabelEncoder()
ndegree=LabelEncoder()
ds['Ncom']=ncompany.fit_transform(ds[['company']])
ds['NJ']=njoB.fit_transform(ds[['job']])
ds['ND']=ndegree.fit_transform(ds[['degree']])
print(ds)
Y=ds[['salary_more_then_100k']]
NDS=ds.drop(['company','job','degree','salary_more_then_100k'],axis='columns')
print(NDS)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
dt=DecisionTreeClassifier()

a,b,c,d=train_test_split(NDS,Y,test_size=0.3)
dt.fit(a,c)
rv=dt.score(b,d)
print(rv)