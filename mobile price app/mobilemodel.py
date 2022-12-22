import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import statsmodels.api as sm
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
import pickle

df=pd.read_csv("MobilePrice_data.csv")
df.drop(columns=["Unnamed: 0.1","Unnamed: 0"],inplace=True)
df.drop_duplicates(inplace=True)

df["Price_range"]=np.where((df["Price"]<=15000),"low range",np.where((df["Price"]>15000)&(df["Price"]<=50000),
"medium range",np.where((df["Price"]>50000)&(df["Price"]<=90000),"High range","very high range")))

cat_col=df.dtypes[df.dtypes==object].index
num_col=df.dtypes[df.dtypes!=object].index

df["RAM"].replace(64,4,inplace=True)
df["Storage"].replace(4,64,inplace=True)
df["Storage"].replace(8,128,inplace=True)
df["Storage"].replace(16,64,inplace=True)
df["Camera_Pixel"].replace(2,64,inplace=True)
df["Camera_Pixel"].replace(5,12,inplace=True)
df["Selfie_cam_pixel"].replace(108,32,inplace=True)


def out_treat(x):
    x=x.clip(upper=x.quantile(0.96))
    x=x.clip(lower=x.quantile(0.05))
    return x
    
df[num_col]=df[num_col].apply(out_treat)

lb=LabelEncoder()
for i in cat_col:
    df[i]=lb.fit_transform(df[i])


x=df.drop(columns=["Price","Colour","Name"])
y=df["Price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.transform(x_test)


lin_reg= LinearRegression()
lasso = Lasso(alpha=0.01)
ridge = Ridge(alpha=0.01)

poly=PolynomialFeatures(degree=2)
poly_reg = LinearRegression()

dec_tree_reg = DecisionTreeRegressor(criterion='poisson',max_depth=7,min_samples_leaf=5,min_samples_split=20)
rand_for_reg = RandomForestRegressor(criterion='squared_error',n_estimators=100,max_depth=10,min_samples_split=15)

xgboost =XGBRegressor(n_estimators=150,max_depth=4,reg_lambda=0.3,eta=0.4,eval_metric='rmse',gamma=0.1,objectives='reg:squarederror',
                random_state=0,reg_alpha=0)


adrf=RandomForestRegressor(criterion="poisson" ,n_estimators=100 ,max_depth=13 ,min_samples_split=20 ,min_samples_leaf=5 ,
                        bootstrap=True,oob_score=True)
ada_reg = AdaBoostRegressor(base_estimator=adrf,n_estimators=100)


lin_reg = lin_reg.fit(x_train_sc,y_train)
lasso = lasso.fit(x_train_sc,y_train)
ridge = ridge.fit(x_train_sc,y_train)
poly_reg = poly_reg.fit(x_train_sc,y_train)
dec_tree_reg = dec_tree_reg.fit(x_train_sc,y_train)
rand_for_reg = rand_for_reg.fit(x_train_sc,y_train)
xgboost = xgboost.fit(x_train_sc,y_train)
ada_reg = ada_reg.fit(x_train_sc,y_train)

pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(lasso,open('lasso_model.pkl','wb'))
pickle.dump(ridge,open('ridge_model.pkl','wb'))
pickle.dump(dec_tree_reg,open('dt_model.pkl','wb'))
pickle.dump(rand_for_reg,open('rf_model.pkl','wb'))
pickle.dump(poly_reg,open('polynomial_model.pkl','wb'))
pickle.dump(xgboost,open('xgboost_model.pkl','wb'))
pickle.dump(ada_reg,open('adaboost_model.pkl','wb'))

