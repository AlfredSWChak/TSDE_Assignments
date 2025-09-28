import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def read_csv(filename) -> str:
    output = pd.read_csv(filename, header = 0)
    return output
    
part1Data = read_csv('../2/data_tsde_assignment_2_part_1.csv')
part2Data = read_csv('../2/data_tsde_assignment_2_part_2.csv')

# brief review of the data sets
# print(part1Data)
# print(part2Data)

'''
Part I: Forecasting, Parameter Estimation, and Model Selection
'''

print('Part I:')

# Question 1

quarterly_growth_rates = part1Data['GDP_QGR']
quarterly_name = part1Data['obs']
fourth_quarter_name = [quarter for quarter in quarterly_name if 'Q4' in quarter]

graphTitle = 'Dutch GDP quarterly growth rates'
fileName = '1_gdp'

plt.figure(figsize=(10,4))    
plt.plot(quarterly_name, quarterly_growth_rates, linewidth = 1, color = 'blue')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(graphTitle, fontweight='bold')
plt.xticks(quarterly_name, minor=True)
plt.xticks(fourth_quarter_name, minor=False, rotation=45)
plt.grid(True, axis='x', linestyle='--', linewidth=0.5, which='minor', color='grey')
plt.grid(True, axis='x', linestyle='--', linewidth=0.8, which='major', color='black')
plt.xlabel('Time')
plt.ylabel('QGR')
plt.savefig(f'../2/figures/{fileName}.jpeg', dpi=300)
plt.show()

def sacf(input_data, lag, graphTitle, fileName):
    result = []
    
    for i in range(1, lag+1):
        x = pd.Series(input_data)
        result.append(x.autocorr(lag = i))
    
    lags = np.arange(1,lag+1,1)
    
    plt.figure(figsize=(10,4))
    plt.bar(lags, result, color='blue', edgecolor='black')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title(graphTitle, fontweight='bold')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.savefig(f'../2/figures/{fileName}.jpeg', dpi=300)
    plt.show()
    
    return result

input_lags = 12

graphTitle = 'Sample ACF of Dutch GDP quarterly growth rates'
fileName = '1_sacf'

gdp_sacf = sacf(quarterly_growth_rates, input_lags, graphTitle, fileName)

# Question 2

def generateMatrixX(input_y, p):
    numberOfRows = len(input_y) - p
    result_matrix = []
    
    for rowNumber in range(0,numberOfRows):
        row = []
        row.append(1)
        
        for t in range(0, p):
            row.append(input_y[p + rowNumber - t - 1])
    
        result_matrix.append(row)
     
    return pd.DataFrame(result_matrix)

def getEstimatedBeta(input_matrix_X, input_vector_y):
    # calculate the inverse matrix (X′ * X)^{−1}
    df = input_matrix_X.T @ input_matrix_X
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                      index=df.columns,   # rows = original columns
                      columns=df.index) 
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    result_beta = df_inv @ input_matrix_X.T @ input_vector_y
    
    return result_beta

def runRegressionModel(input_y, input_p):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y[input_p:])
    vector_y = vector_y.reset_index(drop=True)
    # convert y to matrix X
    matrix_X = generateMatrixX(input_y, input_p)
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    residuals = vector_y - estimate_y
    
    temp_df = matrix_X.copy()
    temp_df['y'] = vector_y
    temp_df['hat_y'] = estimate_y
    temp_df['res'] = residuals

    return estimate_beta, estimate_y, residuals

# compute Bayesian information criterion
def getBIC(t, k, residuals):
    ssr_k = 0
    
    for res in residuals:
        ssr_k += res ** 2
    
    bic = t * np.log(ssr_k/t) + k * np.log(t)
    
    return bic

max_p = 4
estimate_phi_list = []
estimate_residuals_list = []
bic_list = []

# estimate with a maximum p up to 4 lags
for iterateP in range(1, max_p+1):
    # estimate an AR(p) model with intercept for the given data
    estimate_phi, hat_y, hat_residuals = runRegressionModel(quarterly_growth_rates, iterateP)
    
    lengthOfSeries = len(quarterly_growth_rates)
    k = iterateP + 1
    
    this_bic = getBIC(lengthOfSeries, k, hat_residuals.squeeze())
    estimate_phi_list.append(estimate_phi)
    estimate_residuals_list.append(hat_residuals)
    bic_list.append(float(this_bic))
    
min_bic = round(np.min(bic_list),3)
final_p = bic_list.index(np.min(bic_list)) + 1

print(f'The final estimate of (p) is {final_p} with the lowest value of the information criterion which is {min_bic}.')

# Question 3

# produce forecasts up to 2 years ahead
final_phi = estimate_phi_list[final_p - 1]
final_residuals = estimate_residuals_list[final_p - 1]
h_step_ahead = 8

def getQuarterlyName(input_x_list):
    result = ''
    lastName = input_x_list[-1]
    lastYear = lastName[0:4]
    lastQuarter = lastName[-1]
    
    if int(lastQuarter) == 4:
        thisYear = int(lastYear) + 1
        thisQuarter = 1
        result = f'{thisYear}Q{thisQuarter}'
    else:
        thisYear = lastYear
        thisQuarter = int(lastQuarter) + 1
        result = f'{thisYear}Q{thisQuarter}'
        
    return result

for step in range(1,h_step_ahead+1):
    row = []
    row.append(1)
    
    for t in range(0, final_p):
        row.append(float(quarterly_growth_rates[len(quarterly_growth_rates) - t - 1]))
    
    forecast_result = pd.DataFrame([row]) @ final_phi
    quarterly_growth_rates = pd.concat([quarterly_growth_rates, pd.Series([forecast_result.iloc[0, 0]])], ignore_index=True)
    
    forecast_result_name = getQuarterlyName(list(quarterly_name))
    quarterly_name = pd.concat([quarterly_name, pd.Series([forecast_result_name])], ignore_index=True)       

fourth_quarter_name = [quarter for quarter in quarterly_name if 'Q4' in quarter]

graphTitle = 'Dutch GDP quarterly growth rates forecasts up to 2 years ahead'
fileName = '3_forecasts'

plt.figure(figsize=(10,4))    
plt.plot(quarterly_name[:-8], quarterly_growth_rates[:-8], linewidth = 1, color = 'blue')
plt.plot(quarterly_name[-9:],quarterly_growth_rates[-9:], linewidth = 1, color = 'red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(graphTitle, fontweight='bold')
plt.xticks(quarterly_name, minor=True)
plt.xticks(fourth_quarter_name, minor=False, rotation=45)
plt.grid(True, axis='x', linestyle='--', linewidth=0.5, which='minor', color='grey')
plt.grid(True, axis='x', linestyle='--', linewidth=0.8, which='major', color='black')
plt.xlabel('Time')
plt.ylabel('QGR')
plt.savefig(f'../2/figures/{fileName}.jpeg', dpi=300)
plt.show()

# Question 4

# estimate the variance of residuals
res_variance = np.var(final_residuals, axis=0)

# compute the variance of error terms
sumOfPhi = 0
for j in range(0, h_step_ahead):
    sumOfPhi += final_phi.loc[1] ** (2*j)

error_h_step_ahead_error_variance = res_variance * sumOfPhi

# compute the confidence interval
alpha = 0.05
CI_upper_list = []
CI_lower_list = []
CI_upper_list.append(list(quarterly_growth_rates)[-9])
CI_lower_list.append(list(quarterly_growth_rates)[-9])

for step in range(1,h_step_ahead+1):
    forecastValue = list(quarterly_growth_rates)[-(h_step_ahead+1-step)]
    confidenceInterval_upperBound = forecastValue + norm.ppf(1 - alpha/2) * math.sqrt(error_h_step_ahead_error_variance.iloc[0])
    CI_upper_list.append(confidenceInterval_upperBound)
    confidenceInterval_lowerBound = forecastValue - norm.ppf(1 - alpha/2) * math.sqrt(error_h_step_ahead_error_variance.iloc[0])
    CI_lower_list.append(confidenceInterval_lowerBound)
    
graphTitle = 'Dutch GDP quarterly growth rates forecasts up to 2 years ahead'
fileName = '4_forecasts_CI'

plt.figure(figsize=(10,4))    
plt.plot(quarterly_name[:-8], quarterly_growth_rates[:-8], linewidth = 1, color = 'blue')
plt.plot(quarterly_name[-9:],quarterly_growth_rates[-9:], linewidth = 1, color = 'red')
plt.plot(quarterly_name[-9:],CI_upper_list, linestyle = '--', linewidth = 1, color = 'red')
plt.plot(quarterly_name[-9:],CI_lower_list, linestyle = '--', linewidth = 1, color = 'red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(graphTitle, fontweight='bold')
plt.xticks(quarterly_name, minor=True)
plt.xticks(fourth_quarter_name, minor=False, rotation=45)
plt.grid(True, axis='x', linestyle='--', linewidth=0.5, which='minor', color='grey')
plt.grid(True, axis='x', linestyle='--', linewidth=0.8, which='major', color='black')
plt.xlabel('Time')
plt.ylabel('QGR')
plt.savefig(f'../2/figures/{fileName}.jpeg', dpi=300)
plt.show()

'''
Part II: Impulse Response Functions, Autoregressive Distributed Lag Models, and Granger Causality
'''

print('Part II:')

# Question 1

quarterly_growth_rates_part2 = part2Data['GDP_QGR']
quarterly_unemployment_rates = part2Data['UN_RATE']
quarterly_name = part2Data['obs']
fourth_quarter_name = [quarter for quarter in quarterly_name if 'Q4' in quarter]

# Plot the graphs

graphTitle = 'Dutch quarterly unemployment rates with GDP growth rates'
fileName = '5_unr_gdp'

plt.figure(figsize=(10,4))    
plt.plot(quarterly_name, quarterly_unemployment_rates, 
         linewidth = 1, color = 'green', label='Unemployment Rate')
plt.plot(quarterly_name, quarterly_growth_rates_part2, 
         linewidth = 1, color = 'blue', label='GDP Growth Unemployment Rate')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(graphTitle, fontweight='bold')
plt.xticks(quarterly_name, minor=True)
plt.xticks(fourth_quarter_name, minor=False, rotation=45)
plt.grid(True, axis='x', which='minor', 
         linestyle='--', linewidth=0.5, color='grey')
plt.grid(True, axis='x', which='major', 
         linestyle='--', linewidth=0.8, color='black')
plt.xlabel('Time')
plt.ylabel('Rates')
plt.legend()
plt.savefig(f'../2/figures/{fileName}.jpeg', dpi=300)
plt.show()

# compute Akaike information criterion
def getAIC(t, k, residuals):
    ssr_k = 0
    
    for res in residuals:
        ssr_k += res ** 2
    
    aic = t * np.log(ssr_k/t) + k * 2
    
    return aic

def getBestP(input_list):
    min_value = round(np.min(input_list),3)
    final_p = input_list.index(np.min(input_list)) + 1
    return min_value, final_p

# Estimate AR model
max_p = 4  # maximum lags
max_q = 4  # maximum lags
aic_list = []

for iterateP in range(1, max_p+1):
    # estimate an AR(p) model with intercept for the given data
    estimate_phi, hat_y, hat_residuals = runRegressionModel(quarterly_growth_rates_part2, iterateP)

    lengthOfSeries = len(quarterly_growth_rates_part2)
    k = iterateP + 1
    this_aic = getAIC(lengthOfSeries, k, hat_residuals.squeeze())
    
    aic_list.append(float(this_aic))

aic_list_ADL = []

def generateMatrixX_ADLModel(input_y, input_x, p, q):
    numberOfRows = min(len(input_y) - p, len(input_x) - q)
    extra = max(q-p, 0)
    result_matrix = []
    
    for rowNumber in range(0,numberOfRows):
        row = []
        display_Row =[]
        row.append(1)
        display_Row.append(1)
        
        for t in range(0, p):
            row.append(input_y[extra + p + rowNumber - t - 1])
            display_Row.append(extra + p + rowNumber - t - 1)
            
        for t in range(0, q+1):
            row.append(input_x[extra + p + rowNumber - t])
            display_Row.append(extra + p + rowNumber - t)
    
        result_matrix.append(row)
     
    return pd.DataFrame(result_matrix)

def runRegressionModel_ADL(input_y, input_x, input_p, input_q):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y[max(input_p, input_q):])
    vector_y = vector_y.reset_index(drop=True)
    # convert y to matrix X
    matrix_X = generateMatrixX_ADLModel(input_y, input_x, input_p, input_q)
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    residuals = vector_y - estimate_y
    
    temp_df = matrix_X.copy()
    temp_df['y'] = vector_y
    temp_df['hat_y'] = estimate_y
    temp_df['res'] = residuals

    return estimate_beta, estimate_y, residuals

def getBestP_ADL(input_list):
    min_value = round(np.min(input_list),3)
    final_p = int(input_list.index(np.min(input_list))/4) + 1
    final_q = (input_list.index(np.min(input_list)) % 4) + 1
    return min_value, final_p, final_q

# Estimate ADL + AR model
for iterateP in range(1, max_p+1):
    for iterateQ in range(1, max_q+1):
        # estimate an AR(p) model with intercept for the given data
        estimate_phi, hat_y, hat_residuals = runRegressionModel_ADL(quarterly_unemployment_rates, quarterly_growth_rates_part2, iterateP, iterateQ)

        lengthOfSeries = len(quarterly_unemployment_rates)
        k = iterateP + 1
        this_aic = getAIC(lengthOfSeries, k, hat_residuals.squeeze())
        
        aic_list_ADL.append(float(this_aic))

# Use AIC for model selection
min_aic, final_p = getBestP(aic_list)

print('For AR(p) model of the GDP growth rate:')
print(f'The final estimate of (p) is {final_p} with the lowest value of the information criterion which is {min_aic}.')


min_aic_ADL, final_p_ADL, final_q_ADL = getBestP_ADL(aic_list_ADL)

print('For ADL(p,q) model for the unemployment rate:')
print(f'The final estimate of (p,q) is ({final_p_ADL},{final_q_ADL}) with the lowest value of the information criterion which is {min_aic_ADL}.')