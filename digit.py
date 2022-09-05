import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.datasets import load_digits;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
import joblib;
ds=load_digits();
print(dir(ds));
# plt.gray();
# for i in range(5):
#   plt.matshow(ds.images[i]);
#   plt.show()

x_test,x_train,y_test,y_train =train_test_split(ds.data,ds.target,test_size=0.8);
model=LogisticRegression();
model.fit(x_train,y_train);
rv=model.score(x_test,y_test);
pr=model.predict(x_test);
# plt.bar(x_test,y_test);
# plt.show();
clf=joblib.dump(model,'pr.pkl')
print(rv);


