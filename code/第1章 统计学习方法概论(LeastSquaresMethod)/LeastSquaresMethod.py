# Script based python 3.7

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)

#残差
def residuals_func(p,x,y):
    return fit_func(p,x) - y

# 十个点
x = np.linspace(0,1,10)
x_point = np.linspace(0,1,1000)

# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y  = [np.random.normal(0,0.1)+y1 for y1 in y_]

def fitting(M=0):
    """
    M 为多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func,p_init,args=(x,y))
    print("Fitting Parameters:",p_lsq[0])

    # 可视化
    plt.plot(x_point,real_func(x_point),label='real')
    plt.plot(x_point,fit_func(p_lsq[0],x_point),label='fitted curve')
    plt.plot(x,y,'bo',label='noise')
    plt.legend()
    plt.show()
    return p_lsq;

m = 9
#p_lsq_M = fitting(M = m)

regularization = 0.0001

def residuals_func_regularization(p,x,y):
    ret = fit_func(p,x) - y
    ret = np.append(ret,np.sqrt(0.5*regularization*np.square(p)))
    return ret

# 最小二乘法,加正则化项
p_init = np.random.rand(m+1)
p_lsq = leastsq(residuals_func,p_init,args=(x,y))
p_lsq_regularization = leastsq(residuals_func_regularization,p_init,args=(x,y))

plt.plot(x_point,real_func(x_point),label='real')
plt.plot(x_point,fit_func(p_lsq[0],x_point),label='fitted curve')
plt.plot(x_point,fit_func(p_lsq_regularization[0],x_point),label='regulation')
plt.plot(x,y,'bo',label='noise')
plt.legend()
plt.show()
