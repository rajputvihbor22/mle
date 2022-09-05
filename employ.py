

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np;

df=pd.read_csv("C:\\Users\\hp\\Desktop\\cssv file\\HR_comma_sep.csv");
# print(dir(df));
Left=df[df.left==1];
Ret=df[df.left==0];
dfn=df.groupby('left').mean()
# print(dfn.head());


x=pd.crosstab(df.salary,df.left).plot(kind='bar');
# barplot = x.plot.bar(rot=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression();
nds=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']];
# print(nds.head());
dumm=pd.get_dummies(df.salary,prefix="salary")
# print(dumm);
DSn=pd.concat([nds,dumm],axis='columns')
# print(DSn.head());

DSn.drop('salary',axis='columns',inplace=True)
print(DSn)
input=DSn
Y=df.left;
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,y_test=train_test_split(input,Y,test_size=0.3);

model.fit(X_train,Y_train);
print(model.score(X_test,y_test));





