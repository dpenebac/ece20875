'''
from	sklearn	import	linear_model,	from	sklearn.metrics	import	
mean_squared_error,	r2_score

regr	=	linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train,y_train)
regr.coef_
regr.intercept_	
y_pred	=	regr.predict(X_test)
r2_score(y_true,y_pred)


from	sklearn.linear_model	
import	LogisticRegression

from	sklearn	import	metrics

logreg	=	LogisticRegression()
logreg.fit(X_train, y_train)
y_pred	=	logreg.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
'''

'''
1. You want to install sensors on the bridges to estimate overall traffic across all the bridges. 
But you only have enough budget to install sensors on three of the four bridges. 
Which bridges should you install the sensors on to get the best prediction of overall traffic?

2. The city administration is cracking down on helmet laws, and wants to deploy police officers on 
days with high traffic to hand out citations. 
Can they use the next day's weather forecast to predict the number of bicyclists that day?

3. Can you use this data to predict whether it is raining based on the number of bicyclists on the 
bridges?
'''

from math import exp
import math
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.utils.validation import column_or_1d

'''
1. You want to install sensors on the bridges to estimate overall traffic across all the bridges. 
But you only have enough budget to install sensors on three of the four bridges. 
Which bridges should you install the sensors on to get the best prediction of overall traffic?
'''

def prob1(data, p):
    X = data[[p]]
    y = data[['Total']]

    X = X.to_numpy()
    y = y.to_numpy()

    for x in X:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    for x in y:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    Y = y
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.20, random_state=101)

    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(start = -1, stop = 2, num = 51)
    #lmbda = [1,100]

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE)) #fill in
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print(p)
    print("Coef, Intercept", model_best.coef_, model_best.intercept_)
    y_pred = model_best.predict(X_test)
    print("R2, score", r2_score(y_test, y_pred))
    print()

    plt.scatter(y_test, y_pred)
    x = y_test.flatten().astype(float)
    y = y_pred.flatten().astype(float)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)

    #formatting
    plt.xlabel('{} Bike Traffic'.format(p))
    plt.ylabel('Total Bike Traffic')
    plt.title(str(p) + " Vs. Total")

    plt.show()

    return (r2_score(y_test, y_pred))

def prob12(data, remaining):

    X = data[remaining]
    y = data[['Total']]

    X = X.to_numpy()
    y = y.to_numpy()

    for x in X:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    for x in y:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.20, random_state=101)

    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(start = -1, stop = 2, num = 51)
    #lmbda = [1,100]

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE)) #fill in
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print("Coef, Intercept", model_best.coef_, model_best.intercept_)
    y_pred = model_best.predict(X_test)
    print("R2, score", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred)
    x = y_test.flatten().astype(float)
    y = y_pred.flatten().astype(float)
    m, b = np.polyfit(x, y, 1)
    print(m, b)
    plt.plot(x, m*x + b)

    #formatting
    plt.xlabel("Remaining Bridge Bike Traffic")
    plt.ylabel('Total Bike Traffic')
    plt.title("Remaining Vs. Total")
    
    plt.show()

    return

def prob13(data):

    X = data[["Brooklyn","Manhattan","Williamsburg","Queensboro"]]
    y = data[['Total']]

    X = X.to_numpy()
    y = y.to_numpy()

    for x in X:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    for x in y:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.20, random_state=101)

    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(start = -1, stop = 2, num = 51)
    #lmbda = [1,100]

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE)) #fill in
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print("Coef, Intercept", model_best.coef_, model_best.intercept_)
    y_pred = model_best.predict(X_test)
    print("R2, score", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred)
    x = y_test.flatten().astype(float)
    y = y_pred.flatten().astype(float)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)

    #formatting
    plt.xlabel("All")
    plt.ylabel('Total')
    plt.title("All Vs. Total")
    
    plt.show()

    return

'''
2. The city administration is cracking down on helmet laws, and wants to deploy police officers on 
days with high traffic to hand out citations. 
Can they use the next day's weather forecast to predict the number of bicyclists that day?
'''

def prob2(data):

    X = data[['High', 'Low', 'Precipitation']]
    y = data[['Total']]

    X = X.to_numpy()
    y = y.to_numpy()

    for x in X:
        for i in range(len(x)):
            if x[i] == 'T':
                x[i] = 0
            elif type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(' (S)', '')
                x[i] = float(x[i]) * 10 #change this
            else:
                x[i] = float(x[i])
    
    for x in y:
        for i in range(len(x)):
            if type(x[i]).__name__ == 'str':
                x[i] = x[i].replace(',', '')
                x[i] = float(x[i])
    
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.20, random_state=101)

    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(start = -1, stop = 2, num = 51)
    #lmbda = [1,100]

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE)) #fill in
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print("Coef, Intercept", model_best.coef_, model_best.intercept_)
    y_pred = model_best.predict(X_test)
    print("R2, score", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred)
    x = y_test.flatten().astype(float)
    y = y_pred.flatten().astype(float)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)

    #formatting
    plt.xlabel("Weather Forecast")
    plt.ylabel('Total Bike Traffic')
    plt.title("Weather Vs. Total Traffic")
    
    plt.show()

    return

def normalize_train(X_train):

    #fill in
    #subtract column average from each feature sample
    #divide each feature sample by the column standard deviation

    mean = []
    std = []
    for c in X_train.T: #for column in X_train
        cAvg = np.mean(c) #column average
        cSD = np.std(c, ddof=1) #column standard deviation
        mean.append(cAvg)
        std.append(cSD)
        for i in range(len(c)):
            c[i] = c[i] - cAvg
            c[i] = c[i] / cSD

    X = X_train
        
    return X, mean, std

def normalize_test(X_test, trn_mean, trn_std):

    #fill in
    #same as normalize train except do not calculate cavg and csd just use the mean and std given
    counter = 0
    for c in X_test.T:
        for i in range(len(c)):
            c[i] = c[i] - trn_mean[counter]
            c[i] = c[i] / trn_std[counter]
        counter += 1

    X = X_test
    
    return X

def train_model(X,y,l):

    #fill in
    model = linear_model.Ridge(alpha=l, fit_intercept=True)
    model.fit(X,y)

    return model

def train_model_lasso(X,y,l):

    #fill in
    model = linear_model.Lasso(alpha=l, fit_intercept=True)
    model.fit(X,y)

    return model

def error(X,y,model):

    #Fill in
    ypred = model.predict(X)
    mse = mean_squared_error(y, ypred)

    return mse

'''
3. Can you use this data to predict whether it is raining based on the number of bicyclists on the 
bridges?
'''
def prob3(data):

    X = data[['Total']]
    y = data[['Precipitation']]

    X = X.to_numpy()
    y = y.to_numpy()

    for x in X:
        for i in range(len(x)):
            x[i] = float(x[i].replace(',', ''))

    for j in y:
        for i in range(len(j)):
            if j[i] == 'T':
                j[i] = 0
            elif '(S)' in j[i]:
                j[i] = 1 #snow
            else:
                j[i] = float(j[i])
    
    for j in y:
        for i in range(len(j)):
            if j[i] > 0:
                j[i] = 1
            else:
                j[i] = 0
    
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.20, random_state=101)

    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    y_train = column_or_1d(y_train) #idk compiler telling me to do this
    y_train=y_train.astype('int') #compiler
    y_test=y_test.astype('int') #compiler
    
    '''j = np.arange(0.01, 0.05, 0.01)
    for i in j:
        logreg = LogisticRegression(C=i)
        model = logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        plt.scatter(X_test, y_pred) 
        plt.scatter(X_test, model.predict_proba(X_test)[:,1])'''

    logreg = LogisticRegression(C=1)
    model = logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    plt.scatter(X_test, y_pred)

    #just plot graph using equation
    x = np.linspace(float(min(X_test)),float(max(X_test)),100)
    m = model.coef_[0][0] #since coef < 1, precipitation decreases the odds of bikers??
    b = model.intercept_[0]
    print(m,b)
    print(min(X_test), max(X_test))
    a = m * x + b
    e = np.exp(-a)
    y = 1 / (1 + np.exp(-a))
    plt.plot(x, y)

    #formatting
    plt.xlabel("Amount of Bikers")
    plt.ylabel('Probability of Precipitation')
    plt.title("Total Bikers vs Precipitation")

    plt.show()

    print("Score", logreg.score(X_test, y_test))
    
    return 

if __name__ == "__main__":
    data = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv", sep = ",")
    data.columns = ["Date", "Day", "High", "Low", "Precipitation",
                    "Brooklyn","Manhattan","Williamsburg","Queensboro",
                    "Total"]

    print("PROBLEM 1")
    r2 = {}
    r2['Brooklyn'] = prob1(data, 'Brooklyn')
    r2['Manhattan'] = prob1(data, 'Manhattan')
    r2['Williamsburg'] = prob1(data, 'Williamsburg')
    r2['Queensboro'] = prob1(data, 'Queensboro')
    
    print(r2)

    d = r2
    min_val = min(d.values())
    remaining = d.keys() - (k for k, v in d.items() if v == min_val) #subtracts minimum key from dictionary keys
    print("Remaining", remaining)

    prob12(data, remaining) #plot the linear regression of total vs the remaining 3 cities
    prob13(data)

    print("\nPROBLEM 2")
    r2 = prob2(data)

    print("\nPROBLEM 3")
    prob3(data)


    