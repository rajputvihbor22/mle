import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as plt
from sklearn.model_selection import train_test_split
ds=pd.read_csv("C:\\Users\\hp\\Desktop\\cssv file\\homeprices.csv")

print(ds)
model=LinearRegression()
a,b,c,d=train_test_split(ds[['area(sqr ft)']],ds[['price(US $)']],test_size=0.3);

model.fit(b,d);
rv=model.score(a,c)
print(rv);

x=model.predict([[2600]])
print(x)
