import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Question 1

def autoRegressiveProcess(phi, time, whiteNoise):
    x_t = np.arange(0,time,1.0)
    x_0 = 0
    x_t[0] = x_0 * phi + whiteNoise[0]

    for i in range(1,time):
        x_t[i] = x_t[i-1] * phi + whiteNoise[i]

    # print(whiteNoise[:10])
    # print(x_t[:10])

    time_plt = np.arange(0,time,1)
    
    graphTitle = f'Gaussian AR(1) process with phi = {phi}'
    fileName = f'1_AR1_phi_{phi}'
    
    plt.plot(time_plt, x_t, linewidth = 0.5, color = 'red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()

t = 1000

varOfWhiteNoise = 1
new_whiteNoise = np.random.normal(loc=0, scale=varOfWhiteNoise, size=t)

phi_a = 0       # a
phi_b = 0.9     # b
phi_c = 1       # c
phi_d = 1.01    # d

autoRegressiveProcess(phi_a, t, new_whiteNoise)
autoRegressiveProcess(phi_b, t, new_whiteNoise)
autoRegressiveProcess(phi_c, t, new_whiteNoise)
autoRegressiveProcess(phi_d, t, new_whiteNoise)

# Question 2

def read_csv(filename) -> str:
    output = pd.read_csv(filename, header = None)
    return output
    
canvasData = read_csv('../1/data_tsde_assignment_1.csv')

graphTitle = 'Data set from Canvas'
fileName = f'2_data_from_canvas'
    
plt.plot(canvasData.index, canvasData, linewidth = 0.5, color = 'red')
plt.title(graphTitle)
plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
plt.show()

def sacf(input_data, lag, graphTitle, fileName):
    
    result = []
    
    for i in range(1, lag+1):
        x = pd.Series(input_data)
        result.append(x.autocorr(lag = i))
    
    lags = np.arange(1,lag+1,1)
    
    plt.bar(lags, result, color='red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()
    
    return result
        
input_lags = 20
canvasData_list = canvasData.squeeze()

graphTitle = 'Sample ACF for canvas data'
fileName = '2_sacf'

canvasData_sacf = sacf(canvasData_list, input_lags, graphTitle, fileName)

# Question 3

estimate_phi = []

for i in range(1, len(canvasData_sacf)+1):
    estimate_phi.append(canvasData_sacf[i-1] ** (1/i))

mean_estimate_phi = round(sum(estimate_phi)/len(estimate_phi),3)

print(f'The estimated phi is {mean_estimate_phi}.')

estimate_residuals = []

for i in range(1, len(canvasData_list)):
    estimate_residual = canvasData_list[i] - mean_estimate_phi * canvasData_list[i-1]
    estimate_residuals.append(estimate_residual)

mean_estimate_residuals = round(sum(estimate_residuals)/len(estimate_residuals),3)

sum_of_variance = 0.0

for estimate_res in estimate_residuals:
    sum_of_variance += (estimate_res - mean_estimate_residuals) ** 2
    
estimate_residuals_variance = round(sum_of_variance * (1/(len(canvasData_list))),3)

print(f'The estimated variance of residuals is {estimate_residuals_variance}.')

# Question 4

def assumption_check (estimate_residuals, question_number):
    
    # Residuals over time
    t_plt = np.arange(1, len(estimate_residuals)+1, 1)

    graphTitle = 'Residuals over time'
    fileName = f'{question_number}_res_over_time'

    plt.scatter(t_plt, estimate_residuals, color='red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()

    # Histogram of the residuals

    graphTitle = 'Histogram of the residuals'
    fileName = f'{question_number}_hist_res'

    plt.hist(estimate_residuals, color='red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()

    # QQ-plot of the residuals

    normal_samples = np.sort(np.random.normal(loc=0, scale=1, size=len(estimate_residuals)))
    sorted_residuals = np.sort(estimate_residuals)

    graphTitle = 'QQ-Plot of residuals vs normal'
    fileName = f'{question_number}_qq_res'

    plt.scatter(normal_samples, sorted_residuals, color='red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()

    # SACF

    graphTitle = 'SACF of residuals'
    fileName = f'{question_number}_sacf_res'

    residuals_sacf = sacf(estimate_residuals, input_lags, graphTitle, fileName)
    
assumption_check (estimate_residuals, 4)

# Question 5

def autoRegressiveProcess_two(phi_1, phi_2, time, whiteNoise):
    x_t = np.arange(0,time,1.0)
    x_0 = 0
    x_t[0] = x_0 * phi_1 + x_0 * phi_2 + whiteNoise[0]
    x_t[1] = x_t[0] * phi_1 + x_0 * phi_2 + whiteNoise[1]

    for i in range(2,time):
        x_t[i] = x_t[i-1] * phi_1 + x_t[i-2] * phi_2 + whiteNoise[i]

    time_plt = np.arange(0,time,1)
    
    graphTitle = f'Gaussian AR(2) process with phi_1 = {phi_1} and phi_2 = {phi_2}'
    fileName = f'5_AR2_phi_{phi_1}_{phi_2}'
    
    plt.plot(time_plt, x_t, linewidth = 0.5, color = 'red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()

t = 1000

varOfWhiteNoise = 1
new_whiteNoise = np.random.normal(loc=0, scale=varOfWhiteNoise, size=t)

phi_1 = 0.4      # a
phi_2 = 0.45     # a
    
autoRegressiveProcess_two(phi_1, phi_2, t, new_whiteNoise)

phi_1 = 0.5     # b
phi_2 = 0.5     # b

autoRegressiveProcess_two(phi_1, phi_2, t, new_whiteNoise)

# Question 6

# Question 7

phi_1 = 0.4      
phi_2 = 0.45     

estimate_residuals = []

for i in range(2, len(canvasData_list)):
    estimate_residual = canvasData_list[i] - phi_1 * canvasData_list[i-1] - phi_2 * canvasData_list[i-2]
    estimate_residuals.append(estimate_residual)

assumption_check (estimate_residuals, 7)

# Question 8

def movingAverageProcess(theta, time, whiteNoise):
    x_t = np.arange(0,time,1.0)
    wn_0 = 0
    x_t[0] = whiteNoise[0] * theta * wn_0

    for i in range(1,time):
        x_t[i] = whiteNoise[i] + theta * whiteNoise[i-1]

    time_plt = np.arange(0,time,1)
    
    graphTitle = f'MA(1) process with theta = {theta}'
    fileName = f'8_MA1_{theta}'
    
    plt.plot(time_plt, x_t, linewidth = 0.5, color = 'red')
    plt.title(graphTitle)
    plt.savefig(f'../1/figures/{fileName}.jpeg', dpi=300)
    plt.show()

t = 1000

varOfWhiteNoise = 1
new_whiteNoise = np.random.normal(loc=0, scale=varOfWhiteNoise, size=t)

theta_1 = 0.5   # a
    
movingAverageProcess(theta_1, t, new_whiteNoise)

theta_1 = 2     # b
    
movingAverageProcess(theta_1, t, new_whiteNoise)

# Question 9