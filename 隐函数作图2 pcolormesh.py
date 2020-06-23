# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 00:11:41 2020

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
#定义点的数量
n=500

#作点
x=np.linspace(-10,10,500)
y=np.linspace(-10,10,500)

#构造点
X,Y=np.meshgrid(x,y)
Z=np.sin(X+Y)

#作图
plt.pcolormesh(X,Y,Z)
plt.show()