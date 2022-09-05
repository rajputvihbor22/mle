import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
df=pd.read_csv("C:\\Users\\hp\\Desktop\\cssv file\\carprices.csv")
# print(df);

ohe=ColumnTransformer([('encoder]',OneHotEncoder(),[0])],remainder='passthrough')

X= np.array(ohe.fit_transform(input), dtype = np.str)
Y=df['Sell_Price'];
input = X.drop([['Sell_Price']],axis='columns');
# print(Y);
X=X[:,1:];
model=LinearRegression();
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3);
print(input);

model.fit(x_train,y_train);
print(model.score(x_test,y_test));




print(X);


