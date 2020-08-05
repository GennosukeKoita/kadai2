import numpy as np
import random
import math
import matplotlib.pyplot as plt

def sigmoid(num):
    return 1.0 / (1.0 + math.exp(-num))

def forward(x_random, u, s, w, num_input):
    output = 0 # 出力
    for j in range(HIDDEN-1):
        for k in range(num_input):
            u[j+1] += s[j+1][k] * x_random[k]
        u[j+1] = sigmoid(u[j+1])
    
    for j in range(HIDDEN):
    	output += u[j] * w[j]
    
    return sigmoid(output)

def backward(mu, y_random, output, w, u, s, x_random, num_input):
    delta2_r = [0 for j in range(HIDDEN)]
    delta1_r = (y_random - output) * output * (1 - output)
    for j in range(HIDDEN):
        w[j] += mu * delta1_r * u[j]
    
    for j in range(HIDDEN):
        delta2_r[j] = delta1_r * w[j] * u[j] * (1 - u[j])
        for k in range(num_input):
            s[j][k] += mu * delta2_r[j] * x_random[k]
    
    return w, s

def calc_error(output, y_random):
    MSE = 0.0 # 二乗誤差
    MSE = (output - y_random)**2 / 2
    return MSE

def back_propagation(x_random, y_random, s, w, num_input, iter):
    E = 0
    plt_e = []
    plt_x = []
    u = np.array([1, 0, 0]) #隠れ層
    mu = 0.01 #学習係数
    for t in range(iter):
        z = forward(x_random, u, s, w, num_input)
        E = calc_error(z, y_random)
        w, s = backward(mu, y_random, z, w, u, s, x_random, num_input)
        if t % 10 == 0:
            plt_e.append(E)
            plt_x.append(t)
        if t % 10000 == 0:
        	print(t, " / ",iter, " / ", "{0:.7f}".format(E))
    
    return z, plt_e, plt_x

def result_1(x_random, z):
    print("------------------")
    print(x_random[1],"-",x_random[2]," -> {0:.4f}".format(z))

def result_2(x_random, z):
    print("------------------")
    print(x_random[1],"-",x_random[2],"-",x_random[3],"-",x_random[4],"-",x_random[5],"-",x_random[6]," -> {0:.4f}".format(z))
'''
#kadai2_1 XOR Problem
INPUT_1 = 3
HIDDEN = 3
ITER = 300000
plot_E_1 = [] #二乗誤差
plot_X_1 = [] #10回反復
#fig_1 = plt.figure()

print("----------Start----------")
x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) #入力層
n = random.randint(0, len(x)-1) #ランダム選出
y = np.array([0, 1, 1, 0]) #理想の値
s = np.random.normal(0, 0.1, (HIDDEN, INPUT_1)) #入力層〜隠れ層の重み
w = np.random.normal(0, 0.1, HIDDEN) #隠れ層〜出力の重み

z, plot_E_1, plot_X_1 = back_propagation(x[i], y[i], s, w, INPUT_1, ITER)
result_1(x[i], z)
ax = fig_1.add_subplot(1,i+1,1)
plt.plot(plot_X_1, plot_E_1)
print("----------End----------")
plt.show()
'''

#kadai2_2 Mirror Symmetry Detection Problem
HIDDEN = 3
INPUT_2 = 7
ITER = 300000
plot_E_2 = [] #二乗誤差
plot_X_2 = [] #10回反復
print("----------Start----------")
x = np.array([[1,0,0,0,0,0,0] for i in range(64)]) #入力層
y = np.array([0, 1]) #理想の値
s = np.random.normal(0, 0.1, (HIDDEN, INPUT_2)) #入力層〜隠れ層の重み
w = np.random.normal(0, 0.1, HIDDEN) #隠れ層〜出力の重み

#出力が1になる入力を抽出！
x_m_s = []
for i in range(64):
    m_s = format(i, '06b')
    for k in range(len(x[i])-1):
        x[i][k+1] = m_s[k]
    if (x[i][1] == x[i][6]) and (x[i][2] == x[i][5]) and (x[i][3] == x[i][4]):
        x_m_s.append(x[i])
#ここまで

n = random.randint(0, len(x)-1) #ランダム選出

if (x[n] == x_m_s[0]).all() or (x[n] == x_m_s[1]).all() or (x[n] == x_m_s[2]).all() or (x[n] == x_m_s[3]).all() or (x[n] == x_m_s[4]).all() or (x[n] == x_m_s[5]).all() or (x[n] == x_m_s[6]).all() or (x[n] == x_m_s[7]).all():
    z, plot_E_2, plot_X_2 = back_propagation(x[n], y[1], s, w, INPUT_2, ITER)
    result_2(x[n], z)
else:
    z, plot_E_2, plot_X_2 = back_propagation(x[n], y[0], s, w, INPUT_2, ITER)
    result_2(x[n], z)
print("----------End----------")
plt.plot(plot_X_2, plot_E_2)
plt.show()
