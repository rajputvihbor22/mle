import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
print(dir(iris))
print(iris.target_names)
print(iris.feature_names)
# print(iris.data)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
# print(df.head())
print(df[df.target==1].head())
print(df[df.target==0].head())
df['flowerName']=df.target.apply(lamda x: iris)
