import numpy as np
import random
import math
import matplotlib.pyplot as plt

def sigmoid(num):
    return 1.0 / (1.0 + math.exp(-num))

def forward(data_random, u, s, w, hidden, num_input):
    output = 0 # 出力
    for j in range(1, hidden):
        for k in range(num_input):
            u[j] += s[j][k] * data_random[k]
        u[j] = sigmoid(u[j])
    for j in range(hidden):
    	output += u[j] * w[j]
    return u, sigmoid(output)

def backward(mu, t_data_random, output, w, u, s, data_random, hidden, num_input):
    delta2_r = np.zeros((hidden))
    delta1_r = (t_data_random - output) * output * (1 - output)
    for j in range(hidden):
        w[j] += mu * delta1_r * u[j]
        delta2_r[j] = delta1_r * w[j] * u[j] * (1 - u[j])
        for k in range(num_input):
            s[j][k] += mu * delta2_r[j] * data_random[k]
    
    return w, s

def calc_error(output, t_data_random):
    MSE = 0.0 # 二乗誤差
    MSE = pow((output - t_data_random), 2) / 2
    return MSE

def back_propagation(data, t_data, hidden, num_input, iter, weights=None):
    E = 0
    mu = 0.9 #学習係数
    s = np.random.normal(0, 0.1, (hidden, num_input)) #入力層〜隠れ層の重み
    w = np.random.normal(0, 0.1, hidden) #隠れ層〜出力の重み
    index = [i for i in range(64)]
    plt_e = []
    plt_x = []
    for t in range(iter):
        u = [1, 0, 0] #隠れ層
        if len(data) == 4:
            n = random.randint(0, len(data)-1) #ランダム選出
        elif len(data) == 64:
            l = random.choices(index, weights=weights,k=1)
            n = l[0]
        u, z = forward(data[n], u, s, w, hidden, num_input)
        E = calc_error(z, t_data[n])
        w, s = backward(mu, t_data[n], z, w, u, s, data[n], hidden, num_input)
        if t % 10 == 0:
            plt_e.append(E)
            plt_x.append(t)
        if t % 10000 == 0:
            if len(data[n]) == 3:
                print(data[n][1],"-",data[n][2]," -> {0:.4f}".format(z))
            elif len(data[n]) == 7:
                print(data[n][1],"-",data[n][2],"-",data[n][3],"-",data[n][4],"-",data[n][5],"-",data[n][6]," -> {0:.4f}".format(z))
            print(t, " / ",iter, " / ", "{0:.7f}".format(E))
    
    return plt_e, plt_x

def XOR_Problem():
    INPUT_1 = 3
    HIDDEN = 3
    ITER = 10000
    x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) #入力層
    y = np.array([0, 1, 1, 0]) #理想の値
    plot_E_1 = [] #二乗誤差
    plot_X_1 = [] #10回反復

    print("----------Start----------")
    plot_E_1, plot_X_1 = back_propagation(x, y, HIDDEN, INPUT_1, ITER)
    print("----------End----------")
    plt.scatter(plot_X_1, plot_E_1)
    plt.show()

def Mirror_Symmetry_Detection_Problem():
    HIDDEN = 3
    INPUT_2 = 7
    ITER = 300000
    plot_E_2 = [] #二乗誤差
    plot_X_2 = [] #10回反復
    print("----------Start----------")
    x = np.array([[1,0,0,0,0,0,0] for i in range(64)]) #入力層
    y = np.zeros(64) #理想の値
    x_probability = np.zeros(64)

    #出力が1になる入力を抽出！
    for i in range(64):
        m_s = format(i, '06b')
        for k in range(len(x[i])-1):
            x[i][k+1] = m_s[k]
        if (x[i][1] == x[i][6]) and (x[i][2] == x[i][5]) and (x[i][3] == x[i][4]):
            y[i] = 1
            x_probability[i] = 1/16
        else:
            x_probability[i] = 1/112
    #ここまで
    plot_E_2, plot_X_2 = back_propagation(x, y, HIDDEN, INPUT_2, ITER, x_probability)
    print("----------End----------")
    plt.scatter(plot_X_2, plot_E_2)
    plt.show()

XOR_Problem()
#Mirror_Symmetry_Detection_Problem()
