'''
梯度下降的算法实现——线形最小二乘为例
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import sklearn as sk
reg=datasets.make_regression(100,1)
lfit= linear_model.LinearRegression()

X=reg[0]
Y=reg[1]
plt.scatter(X, Y)
plt.show()
lfit.fit(X,Y)
Xt=reg[0].transpose()
coef=lfit.coef_
m,n = X.shape
#初始化系数
A=np.ones((n,1))
At=A.transpose()
b=np.ones((m,1))
#计算 损失函数的梯度
XtX=np.dot(Xt,X)
XtXA=np.dot(XtX,A)
XtY=np.dot(Xt,Y)
grad=XtXA-XtY
step=0.0001
dif=0.001
n=1
th_list=[]
grad_list=[]
while np.linalg.norm(grad, ord = 2) >dif:#梯度的欧几里得范数
    #步长*对应的梯度

    A=A-step*grad
    At=A.transpose()
# 更新梯度
    XtX=np.dot(Xt,X)
    XtXA=np.dot(XtX,A)
    XtY=np.dot(Xt,Y)
    grad=XtXA-XtY
    grad_list.append(grad)
    th_list.append(n)
    n+=1
#计算b
grad_list=list(map(float,grad_list))
plt.plot(pd.DataFrame(th_list),pd.DataFrame(grad_list))
plt.xlabel('th')
plt.ylabel('grad')
plt.show()
b=np.mean(Y)-A*np.mean(X)
print(A)
print(lfit.coef_)
print(lfit.intercept_)
print(b)