import numpy as np
import pandas as pd

#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #for the model parameters in each case. Append the result to paramFits each time.
    file = pd.read_csv(datapath, sep = " ", header = None)
    file.columns = ["x", "y"]
    for n in degrees:
        X = feature_matrix(list(file.x), n)
        B = least_squares(X, file.y)
        paramFits.append(B)

    return paramFits


#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    counter = 0
    counterD = d
    X = np.zeros((len(x),d + 1))
    for i in range(len(x)):
        for j in range(d + 1):
            X[i][j] = x[counter] ** counterD
            counterD -= 1
        counter += 1
        counterD = d

    return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    #B = (X^T * X)^-1 * X^T * y
    XT = np.transpose(X)
    B = np.matmul(np.matmul(np.linalg.inv((np.matmul(XT, X))), XT), y)

    return B

if __name__ == '__main__':
    datapath = 'poly.txt'
    degrees = [2, 4]

    paramFits = main(datapath, degrees)
    print(paramFits)
