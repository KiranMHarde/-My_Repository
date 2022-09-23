import pandas as pd
data=pd.read_csv(r"C:/Users/KIRAN/Downloads/50_Startups (2).csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
SI=SimpleImputer()

SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])


SI_Administration=SI.fit(data[['Administration']])
data['Administration']=SI_Administration.transform(data[['Administration']])


SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])


SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
data['State']=LB.fit_transform(data['State'])

print(data.corr()['Profit'])
data=data.drop('State',axis=1)

import matplotlib.pyplot as plt
plt.plot(data['R&D Spend'],data['Profit'])
plt.xlabel("R&D spend")
plt.ylabel("Profit spend")
plt.show()

plt.plot(data['Profit'],data['Marketing Spend'])
plt.xlabel('Profit spend')
plt.ylabel("Marketing Spend")
plt.show()

X=data.iloc[:,0:4].values
Y=data.iloc[:,-1].values



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
sc=[]
for i in range(2,11):
    regr_2 = DecisionTreeRegressor(max_depth=i)
    regr_2.fit(X_train, Y_train)
    y_pred=regr_2.predict(X_test)
    #y_pred=model.predict(X_test)
    score=r2_score(Y_test,y_pred)
    sc.append(score)

plt.plot(range(2,11),sc)
plt.xlabel("Range of max_depth")
plt.ylabel("Accuracy score")
plt.title("Accuracy score for Decision tree regressor")
plt.show()
from sklearn.ensemble import RandomForestRegressor
sc1=[]
for j in range(2,10):
    regr = RandomForestRegressor(max_depth=j, random_state=0)
    regr.fit(X_train, Y_train)
    r_pred=regr.predict(X_test)
    score1=r2_score(Y_test,r_pred)
    sc1.append(score1)
plt.plot(range(2,10),sc1)
plt.xlabel("Range of max_depth")
plt.ylabel("Accuracy score")
plt.title("Accuracy score for Random forest regressor")
plt.show()

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

t=[]

for l in range(2,11):
    polynomial_features= PolynomialFeatures(degree=l)
    x_poly = polynomial_features.fit_transform(X_train)
    model = LinearRegression()
    model.fit(x_poly, Y_train)
    y_poly_pred = model.predict(x_poly)
    score2=r2_score(y_poly_pred,Y_train)
    t.append(score)
plt.plot(range(2,11),t)
plt.xlabel("Range of Degree")
plt.ylabel("Accuracy score")
plt.title("Accuracy score for Polynomial Regressoin")
plt.show()

from sklearn.neighbors import KNeighborsRegressor
kn_rg = KNeighborsRegressor()
kn_rg.fit(X_train,Y_train)
kn_pred = kn_rg.predict(X_test)
score3= r2_score(Y_test,kn_pred)
print("Prediction by using knn:",score3)


    
