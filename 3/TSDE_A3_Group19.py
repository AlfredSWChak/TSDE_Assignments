import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
import math

# calculation for estimated beta by OLS regression
def getEstimatedBeta(input_matrix_X, input_vector_y):
    # calculate the inverse matrix (X′ * X)^{−1}
    df = input_matrix_X.T @ input_matrix_X
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                      index=df.columns,   # rows = original columns
                      columns=df.index) 
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    result_beta = df_inv @ input_matrix_X.T @ input_vector_y
    
    return result_beta

def runRegressionModel(input_y, input_x):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y)
    # print(vector_y)
    # convert y to matrix X
    matrix_X = pd.DataFrame({'intercept': 1, 'x': input_x})
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    estimate_residuals = vector_y - estimate_y

    return matrix_X, estimate_beta, estimate_y, estimate_residuals

def getStandardError(input_matrix, input_residuals):
    # get the shape of the matrix
    n, k = input_matrix.shape

    # estimate the error variance
    estimated_variance = (input_residuals.T @ input_residuals).iat[0, 0] / (n - k)
    
    df = input_matrix.T @ input_matrix
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                        index=df.columns,   # rows = original columns
                        columns=df.index) 

    # compute the variance-covariance matrix
    varCovMatrix = estimated_variance * df_inv

    # get the standard errors which are the diagonal element
    estimated_standard_error = np.sqrt(np.diag(varCovMatrix))
    
    return estimated_standard_error

def getRSquare(input_residuals_list, true_y_list):
    mean_y = np.average(true_y_list)
    ssr = 0.0
    sst = 0.0
    
    for res, true_y in zip(input_residuals_list, true_y_list):
        ssr += res ** 2
        sst += (true_y - mean_y) ** 2

    rSquare = 1 - (ssr / sst)
    
    return rSquare

def getKernelDensity(input_variables):
    kde = gaussian_kde(np.asarray(input_variables))
    x = np.linspace(min(input_variables), max(input_variables), 200)
    density = kde(x)
    
    return x, density
    

def distributionPlot(input_x_list, input_density_list, input_numberOfSimulations_list, input_xLabel, input_title, input_fileName):
    plt.figure(figsize=(10,6)) 
    
    for i in range(0,len(input_numberOfSimulations_list)):
        plt.plot(input_x_list[i], input_density_list[i], 
                 linestyle='-', linewidth=2,
                 label=f'T={input_numberOfSimulations_list[i]}')
    
    plt.title(f'{input_title}', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel(input_xLabel)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../3/figures/{input_fileName}.jpeg', dpi=300)
    plt.show()
    
t_list = [100, 500, 1000]
numberOfSimulations_B = 10000

beta_x_list = []
beta_density_list = []
t_x_list = [] 
t_density_list = []
rSquare_x_list = []
rSquare_density_list = []

for t in t_list:
    this_beta_list = []
    this_t_list = []
    this_RSquare_list = []

    for i in range(0, numberOfSimulations_B):
        v_t = np.random.normal(loc=0, scale=1, size=t+1)
        w_t = np.random.normal(loc=0, scale=1, size=t+1)
    
        this_y_list = []
        this_y_list.append(0)
        this_x_list = []
        this_x_list.append(0)

        for j in range(1, t+1):
            y_t = this_y_list[j-1] + v_t[j]
            this_y_list.append(y_t)
            x_t = this_x_list[j-1] + w_t[j]
            this_x_list.append(x_t)
        
        matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(this_y_list, this_x_list)
        
        estimate_SE = getStandardError(matrix_X, hat_residuals)
        
        t_statistic = estimate_beta.iat[1,0] / estimate_SE[1]
        
        rSquare = getRSquare(hat_residuals.squeeze(), this_y_list)
        
        this_beta_list.append(estimate_beta.iat[1, 0])
        this_t_list.append(t_statistic)
        this_RSquare_list.append(rSquare)
        
    beta_x, beta_density = getKernelDensity(this_beta_list)
    beta_x_list.append(beta_x)
    beta_density_list.append(beta_density)
    
    t_x, t_density = getKernelDensity(this_t_list)
    t_x_list.append(t_x)
    t_density_list.append(t_density)
    
    rSquare_x, rSquare_density = getKernelDensity(this_RSquare_list)
    rSquare_x_list.append(rSquare_x)
    rSquare_density_list.append(rSquare_density)
          
graphTitle = 'Distribution of Estimated hat_betas'
fileName = '1_MC_pdf_beta'
xName = 'beta'
distributionPlot(beta_x_list, beta_density_list, t_list, xName, graphTitle, fileName)

graphTitle = 'Distribution of Estimated t-statistics'
fileName = '1_MC_pdf_t'
xName = 't'
distributionPlot(t_x_list, t_density_list, t_list, xName, graphTitle, fileName)

graphTitle = 'Distribution of Estimated R^2'
fileName = '1_MC_pdf_RSquare'
xName = 'R^2'
distributionPlot(rSquare_x_list, rSquare_density_list, t_list, xName, graphTitle, fileName)