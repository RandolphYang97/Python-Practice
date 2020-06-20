#PCA
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

iris=datasets.load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data.head()
target=iris.target

label_dict = {0: 'Iris-Setosa',
              1: 'Iris-Versicolor',
              2: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}

iris_data['class num']=target
for k,v in label_dict.items():
    for x1 in range(len(iris_data['class num'])):
        if iris_data.loc[x1,'class num']== k:
            iris_data.loc[x1,'class']=v


plt.figure(figsize=(10,5))
x=[]

for cnt in range(4):
    plt.subplot(2,2,cnt+1)
    for lab in ('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virgnica'):
        for x1 in range(len(iris_data.loc[:,'class'])):
            if lab==iris_data.loc[x1,'class']:
                x.append(iris_data.iloc[x1,cnt])
        plt.hist(x,
                 label=lab,
                 bins=20,
                 alpha=0.5,)
        x=[]
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right',fancybox=True,fontsize=8)

plt.tight_layout()
plt.show()
'''
 降维或因子分析前进行标准化，
     1.消除量纲对协方差矩阵的影响。
     2.使分布中心E(X)移到原点
     3.缩小或扩大坐标轴，使分布的密度适中
而此处量纲都是cm，只需去中心化。去中心化是为了让数据点聚集到原点附近，也是主成分分析的定义需要
'''
X_ori= np.array(iris_data.iloc[:,0:4])
X_std = np.array(iris_data.iloc[:,0:4]-np.mean(iris_data.iloc[:,0:4], axis=0))
#%%  使用sklearn  
'''
使用sklearn的pca时不需要将数据去中心化，直接输入数据矩阵
'''
from sklearn.decomposition import PCA
y=iris.target
X1=iris.data
pca= PCA(n_components=4)
X_r=pca.fit(X1).transform(X1)
# 解释方差
print(sum(pca.explained_variance_ratio_))
#特征值
pca.components_
pca.singular_values_
print(pca.singular_values_)
pca.components_
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
print(pca.components_)
target_names = iris.target_names
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
#%%   算式计算
#均值
#mean_X = np.mean(X_std, axis=0)
#协方差矩阵    E((f(Xi-E(Xi))(Xj-E(Ej)))   (X-X-mean)T * (X-X-mean)/(n-1) 样本协方差
#cov= (X_std - mean_X).T.dot((X_std - mean_X)) / (X_std.shape[0]-1)
#或者
# 此处应该转置，因为原数据shape为 (150，4)，4是对应方程的4个变量，矩阵(4,150)*矩阵(150,4)才是计算4个变量的协方差矩阵
#cov = np.cov(X_std,rowvar=0)
#计算协方差矩阵的特征值与特征向量：
#eig_vals,eig_vec=np.linalg.eig(cov)
#比较特征值的绝对值大小
U, eig_vals, eig_vec = np.linalg.svd(X_std, full_matrices=False)
eig_sort=[(np.abs(eig_vals[i]),eig_vec[i]) for i in range(len(eig_vals))]
'''
笔记：
1.使用列表嵌套元组结构时，要用lambda 函数。lambda中冒号前面的x是形式参数，
冒号后面表示该形参的表达式。例如 lambda x：x+1  表示lambda函数的形参x接受值后，
返回x+1的结果。
2.key参数表示排序前对列表每个元素进行预处理。
举例：
a=[5,1,3,4,-1,-4]
a.sort()
a.sort(key=lambda x:x*(-1))
即排序前先将每个元素乘以-1 再排序，相当于降序排序
'''
eig_sort.sort(key=lambda eig_v:eig_v[0], reverse=True)
for i in eig_sort:
    print(i[0])
sum_eigval=sum(eig_vals)
exped_var=[(i/sum_eigval)*100 for i in sorted(eig_vals,reverse=True)]
cum_exped_var=np.cumsum(exped_var)#cumsum 各行累加如：第一行， 第一行+第二行....

#绘图，各成分方差解释和累计解释
plt.figure(figsize=(6, 4))
#plt(x坐标，y坐标，参数)
plt.bar(range(4), exped_var, alpha=0.8, align='center',
            label=' Explained Variance')
plt.step(range(4), cum_exped_var, where='mid',
             label='Cumulative Explained Variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='center right')
plt.tight_layout()
plt.show()
#如上sklearn的设置，取特征值最大的前两个
#vstack，行数增加的堆叠方式  hstack列数增加的堆叠方式,输入元组
# 重排特征向量为4行2列
weight_mat=np.hstack((eig_sort[0][1].reshape(4,1),
                     eig_sort[1][1].reshape(4,1)))
# 最后，将样本点投影到特征向量上
Y=np.dot(X_std,weight_mat)
#绘图
#主成分分析前
X=np.array(iris_data)
y=np.array(iris_data.iloc[:,5])
plt.figure(1,figsize=(6,4))
plt.subplot(2,2,1)
for lab, color in zip(('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virgnica'),
                      ('blue', 'green', 'orange')):  
    plt.scatter(X[y==lab,0],
                X[y==lab,1],
                label=lab,
                c=color)
plt.xlabel('Sepal len')
plt.ylabel('Sepal wid')
plt.legend(loc='best')
plt.tight_layout()

plt.subplot(2,2,2)
for lab, color in zip(('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virgnica'),
                      ('red', 'green', 'blue')):  
    plt.scatter(X[y==lab,2],
                X[y==lab,3],
                label=lab,
                c=color)
plt.xlabel('Petal len')
plt.ylabel('Petal wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#主成分分析后

plt.figure(2,figsize=(6, 4))
for lab, color in zip(('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virgnica'),
                        ('blue', 'green', 'orange')):
     plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=color)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.plot(weight_mat[:,0], weight_mat[:,1])
plt.show()

#%% PCA N=3
from mpl_toolkits.mplot3d import Axes3D
weight_mat=np.hstack((eig_sort[0][1].reshape(4,1),
                     eig_sort[1][1].reshape(4,1),
                     eig_sort[2][1].reshape(4,1)))
Y=np.dot(X_std,weight_mat)
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(Y[:, 0], Y[:, 1],Y[:, 2], c=iris_data.loc[:,'class num'],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])